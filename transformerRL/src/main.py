import os
import tensorflow as tf
import numpy as np
import random

# 导入重构后的模块
from common.config import Config
from data_loader.data_utils import Utils
from trainers.bpr_trainer import train_bpr_and_cluster
from trainers.rl_trainer import RLTrainer
from envs.user_env import User_Enviroment 
from models.dqn import DQNU

# ==================== TF 日志屏蔽设置 ====================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# =======================================================

def main():
    # 1. 初始化配置与工具
    cfg = Config()
    utils = Utils()
    
    # 2. 加载数据
    data_path = "../data/" + cfg.DATA_SET + "_raw"
    print(f">>> [Stage 0] Loading Dataset from {data_path}...")
    
    # 【修改点】接收 item_count
    user_historicals, user_rating_matrix, interactive_length, item_count = utils.load_raw_data_rating(data_path)
    
    # 【重要】动态更新 Config 中的 ITEM_SIZE，防止硬编码导致的越界
    cfg.ITEM_SIZE = item_count
    print(f"    Data Loaded. User Count: {len(user_historicals)}, Item Count (Size): {cfg.ITEM_SIZE}")
    
    # 3. 划分数据集
    train_user_id = []
    test_user_id = []
    val_user_id = []
    
    for user in user_historicals:
        r = random.random()
        if r < 0.8:
            train_user_id.append(user)
        elif r < 0.9:
            val_user_id.append(user)
        else:
            test_user_id.append(user)
    
    print(f"    Train Users: {len(train_user_id)}, Test Users: {len(test_user_id)}")

    # 4. Stage 1: BPR 预训练与聚类
    print(">>> [Stage 1] BPR Pre-training & User Clustering...")
    user_cluster_map, cluster_embs = train_bpr_and_cluster(
        utils, cfg, user_historicals, user_rating_matrix, train_user_id
    )

    # 5. Stage 2: 初始化 RL 环境
    print(">>> [Stage 2] Initializing RL Environments...")
    train_user_history = {u: user_historicals[u] for u in train_user_id}
    
    train_envs = []
    for _ in range(cfg.user_batch_size):
        env = User_Enviroment(
            train_user_history, 
            user_rating_matrix, 
            user_cluster_map, 
            cfg.CLUSTER_NUMS, 
            cfg
        )
        train_envs.append(env)

    # 6. Stage 3: 构建 DQN 模型
    print(">>> [Stage 3] Building DQN Model...")
    dqn_model = DQNU(cfg, cluster_embs, scope='dq1')
    dqn_model.init_model() 

    # 7. Stage 4: 启动 RL 训练循环
    print(">>> [Stage 4] Starting RL Training Loop...")
    trainer = RLTrainer(
        model=dqn_model,
        train_envs=train_envs,
        cfg=cfg,
        train_user_ids=train_user_id,
        test_user_ids=test_user_id,
        user_rating_matrix=user_rating_matrix,
        user_historicals=user_historicals,
        user_cluster_map=user_cluster_map
    )
    
    trainer.run()

if __name__ == "__main__":
    main()