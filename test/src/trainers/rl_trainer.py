import numpy as np
import os
import time
from data_loader.memory import Memory
from common.logger_utils import TrainingLogger
from envs.user_env import get_norm_rating  # 导入我们在 envs 中定义的函数

def compute_batch_ndcg(hit_seqs):
    batch_size, seq_size = hit_seqs.shape
    idcg_seq = np.ones([seq_size,])
    idcg = np.sum(idcg_seq / np.log2(np.arange(2, idcg_seq.size + 2)))
    ndcg_res = []
    for i in range(batch_size):
        dcg = np.sum(hit_seqs[i] / np.log2(np.arange(2, hit_seqs[i].size + 2)))
        ndcg = dcg/idcg
        ndcg_res.append(ndcg)
    return ndcg_res

class RLTrainer:
    def __init__(self, model, train_envs, cfg, train_user_ids, test_user_ids, user_rating_matrix, user_historicals, user_cluster_map):
        self.model = model
        self.train_envs = train_envs
        self.cfg = cfg
        self.train_user_ids = train_user_ids
        self.test_user_ids = test_user_ids
        self.user_rating_matrix = user_rating_matrix
        self.user_historicals = user_historicals
        self.user_cluster_map = user_cluster_map
        
        # 初始化 Replay Buffer 和 Logger
        self.memory = Memory(cfg)
        self.logger = TrainingLogger(cfg)
        self.mem_iter = self.memory.get_record_iter()
        
        # 训练状态追踪
        self.episode_count = 0
        self.highest_hit = 0.0
        self.highest_epoch = 0

    def run(self):
        # 1. 建立实验目录
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        exp_dir = "../experiments/" + self.cfg.DATA_SET + "_" + timestamp
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        log_file_path = os.path.join(exp_dir, "train_log.txt")
        csv_file_path = os.path.join(exp_dir, "train_metrics.csv")
        
        print(f"Logs will be saved to: {log_file_path}")
        
        # 初始化 CSV 表头
        with open(csv_file_path, "w") as f:
            f.write("episode,avg_test_hit,phase1_hit,phase2_hit,avg_ndcg,train_loss\n")

        # 2. 准备训练变量
        total_training_episodes_nb = 500000
        user_batch_size = self.cfg.user_batch_size
        
        # 随机打乱训练用户
        t_train_user_id = self.train_user_ids.copy()
        np.random.shuffle(t_train_user_id)
        
        lhs = 0
        rhs = lhs + user_batch_size
        user_nb = len(t_train_user_id)
        
        # ==================== Main Loop ====================
        while self.episode_count < total_training_episodes_nb:
            
            train_loss = []
            evaluate_point = 5000
            next_evaluate_point = self.episode_count + evaluate_point
            
            # --- Training Phase (收集数据 + 训练) ---
            while self.episode_count < next_evaluate_point:
                self.episode_count += user_batch_size
                
                # 获取当前 Batch 的用户 ID 和 聚类 ID
                batch_users = t_train_user_id[lhs:rhs]
                np_batch_users = np.reshape(np.array([self.user_cluster_map[_u] for _u in batch_users]), [-1,1])
                batch_users_nb = len(batch_users)
                
                # 重置环境
                p_state, n_state = [], []
                for _i in range(batch_users_nb):
                    _p, _n, _ = self.train_envs[_i].reset_env()
                    p_state.append(_p)
                    n_state.append(_n)
                
                ban_items = np.ones(shape=[batch_users_nb, self.cfg.ITEM_SIZE], dtype=np.float32)
                ban_next_items = ban_items.copy()
                
                inner_step = 0
                
                # 用户交互循环 (MAX_STEPS 步)
                for _step in range(self.cfg.MAX_STEPS):
                    # 策略选择逻辑
                    if _step < self.cfg.switch_step:
                        # 通用策略阶段：使用 Cluster ID = CLUSTER_NUMS (即中心 Embedding)
                        acutal_batch_pred_users = np.array([self.cfg.CLUSTER_NUMS] * batch_users_nb, dtype='int32')
                    if _step == self.cfg.switch_step:
                        # 切换点：使用 Discriminator 预测用户归属
                        batch_pred_users, _ = self.model.pred_user(p_state, n_state)
                        acutal_batch_pred_users = batch_pred_users
                    
                    batch_pred_users = acutal_batch_pred_users
                    batch_pred_users = np.reshape(batch_pred_users, [-1,1])
                    
                    # 选择动作 (Items)
                    choosed_items = self.model.choose_batch_action(p_state, n_state, ban_items, batch_pred_users)
                    
                    p_state_next = [[i for i in p_state[j]] for j in range(batch_users_nb)]
                    n_state_next = [[i for i in n_state[j]] for j in range(batch_users_nb)]
                    rewards = []
                    
                    # 执行动作并获取反馈
                    for _j in range(batch_users_nb):
                        user_id = batch_users[_j]
                        item = choosed_items[_j]
                        
                        rating = self.user_rating_matrix[user_id][item]
                        norm_rating = get_norm_rating(rating, 0.0, 5.0)
                        
                        info_gain = self.train_envs[_j].get_info(item, norm_rating)
                        
                        if item in self.user_historicals[user_id] and rating >= self.cfg.boundary_rating:
                            p_state_next[_j].pop(0)
                            p_state_next[_j].append(item)
                        else:
                            n_state_next[_j].pop(0)
                            n_state_next[_j].append(item)
                        
                        ban_next_items[_j][item] = 0.0
                        rewards.append(norm_rating + info_gain)

                    # 存入 Memory
                    for _j in range(batch_users_nb):
                        self.memory.add_record(
                            p_state[_j], n_state[_j], p_state_next[_j], n_state_next[_j], 
                            choosed_items[_j], ban_items[_j], ban_next_items[_j], rewards[_j], 
                            False, np_batch_users[_j], batch_pred_users[_j], inner_step, 
                            self.train_envs[_j].current_distribution
                        )
                    
                    # 更新状态
                    ban_items = ban_next_items.copy()
                    p_state = [[i for i in p_state_next[j]] for j in range(batch_users_nb)]
                    n_state = [[i for i in n_state_next[j]] for j in range(batch_users_nb)]
                    inner_step += 1
                
                # 网络更新 (当 Buffer 足够大时)
                if self.memory.count > self.cfg.START_TRAINING_LIMIT:
                    batch_data = next(self.mem_iter)
                    # 解包 batch_data
                    (b_p, b_n, b_pn, b_nn, b_a, b_ban, b_ban_n, b_r, b_t, b_u, b_pre_u, b_s, b_dist) = batch_data
                    
                    # 更新 Actor-Critic 网络
                    loss = self.model.learn_acnet(b_p, b_n, b_pn, b_nn, b_a, b_ban, b_ban_n, b_r, np.reshape(b_pre_u, [-1,1]))
                    train_loss.append(loss)
                    
                    # 更新 Discriminator 网络
                    self.model.learn_discrimitor(b_p, b_n, np.reshape(b_u, [-1,1]))

                # 循环遍历用户 Batch
                lhs = rhs
                if lhs >= user_nb - 1:
                    lhs = 0
                    np.random.shuffle(t_train_user_id) # 每个 Epoch 重新打乱
                rhs = lhs + user_batch_size

            # --- Evaluation Phase (Test) ---
            print(f"Evaluating at episode {self.episode_count}...")
            current_hit, p1_hit, p2_hit, avg_ndcg, avg_reward = self.evaluate()
            current_loss = np.mean(train_loss) if train_loss else 0.0

            # Logging & Saving
            save_msg = ""
            if self.highest_hit < current_hit:
                self.highest_hit = current_hit
                self.highest_epoch = self.episode_count
                
                # 调用模型自身的 save_model 方法
                save_path = os.path.join(ckpt_dir, "model.ckpt")
                self.model.save_model(save_path)
                save_msg = "!!! Best Model Saved !!!"

            # 打印日志到控制台
            log_str = (
                f"Episode: {self.episode_count}\n"
                f"Hit: {current_hit:.4f} (P1: {p1_hit:.4f}, P2: {p2_hit:.4f})\n"
                f"NDCG: {avg_ndcg:.4f}\n"
                f"Loss: {current_loss:.4f}\n"
                f"{save_msg}\n"
                "--------------------------------------------------\n"
            )
            print(log_str)
            
            # 写入日志文件
            with open(log_file_path, "a") as f:
                f.write(log_str)
            
            # 写入 CSV
            with open(csv_file_path, "a") as f:
                f.write(f"{self.episode_count},{current_hit},{p1_hit},{p2_hit},{avg_ndcg},{current_loss}\n")

            # 更新 Target Network
            self.model.update_target_params()

    def evaluate(self):
        """
        在测试集上评估模型性能
        """
        test_lhs = 0
        test_rhs = self.cfg.user_batch_size
        
        total_hit, total_p1, total_p2, total_ndcg, total_reward = [], [], [], [], []
        
        while test_rhs < len(self.test_user_ids):
            batch_users = self.test_user_ids[test_lhs:test_rhs]
            batch_len = len(batch_users)
            
            # 初始化状态
            p_state = [[0]*self.cfg.HISTORY_SIZE for _ in range(batch_len)]
            n_state = [[0]*self.cfg.HISTORY_SIZE for _ in range(batch_len)]
            ban_items = np.ones([batch_len, self.cfg.ITEM_SIZE], dtype=np.float32)
            
            hits = [0] * batch_len
            p1_hits = [0] * batch_len
            p2_hits = [0] * batch_len
            hit_seq = np.zeros([batch_len, self.cfg.MAX_STEPS])
            rewards = [0.0] * batch_len
            
            for _step in range(self.cfg.MAX_STEPS):
                # 预测阶段
                if _step < self.cfg.switch_step:
                    pred_users = np.array([self.cfg.CLUSTER_NUMS] * batch_len, dtype='int32')
                else:
                    pred_users, _ = self.model.pred_user(p_state, n_state)
                
                pred_users = np.reshape(pred_users, [-1,1])
                
                # 选择动作
                choosed = self.model.choose_batch_action(p_state, n_state, ban_items, pred_users)
                
                for j in range(batch_len):
                    u_id = batch_users[j]
                    item = choosed[j]
                    rating = self.user_rating_matrix[u_id][item]
                    
                    # 检查是否命中 (Rating >= 3.0 且在历史中)
                    if item in self.user_historicals[u_id] and rating >= self.cfg.boundary_rating:
                        p_state[j].pop(0); p_state[j].append(item)
                        hits[j] += 1
                        hit_seq[j][_step] = 1.0
                        
                        if _step < self.cfg.switch_step:
                            p1_hits[j] += 1
                        else:
                            p2_hits[j] += 1
                    else:
                        n_state[j].pop(0); n_state[j].append(item)
                    
                    ban_items[j][item] = 0.0
                    rewards[j] += get_norm_rating(rating, 0.0, 5.0)

            # 计算本 Batch 指标
            total_hit.extend([h / self.cfg.MAX_STEPS for h in hits])
            total_p1.extend([h / self.cfg.switch_step for h in p1_hits])
            total_p2.extend([h / (self.cfg.MAX_STEPS - self.cfg.switch_step) for h in p2_hits])
            total_ndcg.extend(compute_batch_ndcg(hit_seq))
            total_reward.extend(rewards)
            
            test_lhs = test_rhs
            test_rhs += self.cfg.user_batch_size
            
        return np.mean(total_hit), np.mean(total_p1), np.mean(total_p2), np.mean(total_ndcg), np.mean(total_reward)