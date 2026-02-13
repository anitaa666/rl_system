import numpy as np
import random
import time
from sklearn.cluster import KMeans
from models.bpr import BPR_MF  # 导入拆分后的 BPR 模型

def generate_a_negative_sample(user, user_history, item_num):
    j = random.randint(1, item_num)
    while j in user_history[user]:
        j = random.randint(1, item_num)
    return j

def get_cluster_embeddings(embs, train_user_embs, k=10):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(train_user_embs)
    cluster_ids = kmeans.predict(embs)
    return cluster_ids, kmeans.cluster_centers_

def train_bpr_and_cluster(utils, cfg, user_historicals, user_rating_matrix, train_user_id):
    """
    运行 BPR 预训练并执行 K-Means 聚类
    返回：
        user_cluster_map: 用户到聚类ID的映射
        cluster_embs: 聚类中心向量 (包含最后的全局中心)
    """
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " BPR start")
    
    users = [u for u in train_user_id]
    
    # 初始化 BPR 模型
    # 注意：这里我们传入 None 让 BPR 自己创建 session，或者你可以传入全局 sess
    model = BPR_MF(None, len(user_historicals), cfg.ITEM_SIZE, cfg.HIDDEN_LAYER_SIZE, 0.005)
    model.init()
    
    item_count = cfg.ITEM_SIZE
    
    # 开始 BPR 训练循环
    for epoch in range(20):
        np.random.shuffle(users)
        batch_users = []
        batch_items = []
        batch_neg_items = []
        losses = []
        
        for user in users:
            # 筛选评分大于边界的交互
            interactions = [_i for _i in user_historicals[user] if user_rating_matrix[user][_i] >= cfg.boundary_rating]
            user_selected_items_size = len(interactions)
            
            for l, item in enumerate(interactions):
                negative_item = generate_a_negative_sample(user, user_historicals, item_count)
                batch_users.append(user)
                batch_items.append(item)
                batch_neg_items.append(negative_item)

                if len(batch_users) >= 256 or l == user_selected_items_size - 1:
                    # 执行训练步
                    current_loss = model.train(batch_users, batch_items, batch_neg_items)
                    losses.append(current_loss)
                    
                    batch_items = []
                    batch_neg_items = []
                    batch_users = []
        
        print(f"BPR Epoch {epoch+1}/20, Loss: {np.mean(losses):.4f}")

    user_embs = model.get_userembeddings()
    print('!!!!!!! BPR done')
    
    # --- 聚类阶段 ---
    train_user_embs = np.array([user_embs[i] for i in range(len(user_embs)) if i in train_user_id])
    
    user_cluster_nb = cfg.CLUSTER_NUMS
    user_cluster_map, cluster_embs = get_cluster_embeddings(user_embs, train_user_embs, user_cluster_nb)

    # 计算全局平均 User Embedding 作为通用策略的 Task Embedding
    center_user_emb = np.mean(train_user_embs, axis=0)
    cluster_embs = np.row_stack((cluster_embs, center_user_emb))
    
    return user_cluster_map, cluster_embs