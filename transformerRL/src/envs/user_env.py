import numpy as np

# --- 全局辅助函数 (RL Trainer 也会用到) ---
fast_norm_ratings = {}

def get_norm_rating(r, r_small, r_large):
    """
    归一化评分，带缓存机制
    """
    if r in fast_norm_ratings:
        return fast_norm_ratings[r]
    a = 2.0 / (float(r_large) - float(r_small)) # 0.4
    b = - (float(r_large) + float(r_small)) / (float(r_large) - float(r_small)) # -1.0

    reward = a * r + b
    fast_norm_ratings[r] = reward
    return reward


class User_Enviroment(object):
    def __init__(self, user_history, user_rating_matrix, user_cluster_map, cluster_nb, cfg):
        self.user_history = user_history
        self.cfg = cfg  # 保存配置对象
        self.user_rating_matrix = user_rating_matrix
        self.user_cluster_map = user_cluster_map
        self.cluster_nb = cluster_nb

        self.rating_matrix = self.get_init_env()

    def get_init_env(self):
        self.user_nb = len(self.user_history)
        # 修改点：使用 self.cfg.ITEM_SIZE
        self.item_nb = self.cfg.ITEM_SIZE
        self.rating_matrix = np.zeros([self.user_nb, self.item_nb]) + get_norm_rating(0.0, 0.0, 5.0)
        self.user_index = np.arange(0, self.user_nb)
        self.user_cluster_matrix = np.zeros([self.user_nb, self.cluster_nb])

        user_count = 0

        for u in self.user_history:
            items = self.user_history[u]
            for i in items:
                self.rating_matrix[user_count][i] = get_norm_rating(self.user_rating_matrix[u][i], 0.0, 5.0) 
            self.user_cluster_matrix[user_count][self.user_cluster_map[u]] = 1.0
            user_count += 1

        self.init_cluster_p = np.sum(self.user_cluster_matrix, axis=0) / self.user_nb

        self.init_cluster_entropy = -np.sum(self.init_cluster_p * np.log(self.init_cluster_p + 1e-6))
        self.last_distribution = self.init_cluster_p 

        return self.rating_matrix

    def get_info(self, action, feedback):
        pos_users, pos_nb, neg_users, neg_nb, user_positive_cluster_entropy, user_negative_cluster_entropy, info_gain = self.get_pos_and_neg_users_w_feedback(action, feedback)
        
        self.current_distribution = self.last_distribution
        if feedback > 0:
            self.user_index = pos_users
            self.last_state_entropy = user_positive_cluster_entropy
            self.last_distribution = self.user_positive_cluster_distribution
        else:
            self.user_index = neg_users
            self.last_state_entropy = user_negative_cluster_entropy
            self.last_distribution = self.user_negative_cluster_distribution
        
        if info_gain > 0.2:
            info_gain = 0.2
        if info_gain < -0.2:
            info_gain = -0.2
        return info_gain

    def get_pos_and_neg_users_w_feedback(self, action, feedback):
        total_user_nb = float(self.user_index.shape[0]) 
        # 注意：这里的 self.env 在 reset_env 中被赋值为 rating_matrix
        pos_mat = np.where(self.env[:, action] > 0.0)
        
        pos_users = np.intersect1d(self.user_index, pos_mat)
        pos_nb = float(pos_users.shape[0])

        neg_users = np.setdiff1d(self.user_index, pos_mat)

        neg_nb = float(neg_users.shape[0])
        
        # 安全检查，防止浮点误差导致的不相等（虽然这里是整数逻辑）
        # assert (neg_nb + pos_nb == total_user_nb)

        if pos_nb > 0:
            user_positive_cluster_distribution = np.sum(self.user_cluster_matrix[pos_users, :], axis=0) / pos_nb
        else:
            user_positive_cluster_distribution = np.array([1.0/self.cluster_nb] * self.cluster_nb)

        if neg_nb > 0:
            user_negative_cluster_distribution = np.sum(self.user_cluster_matrix[neg_users, :], axis=0) / neg_nb
        else:
            user_negative_cluster_distribution = np.array([1.0/self.cluster_nb] * self.cluster_nb)

        user_positive_cluster_entropy = -np.sum(user_positive_cluster_distribution * np.log(user_positive_cluster_distribution + 1e-6))
        user_negative_cluster_entropy = -np.sum(user_negative_cluster_distribution * np.log(user_negative_cluster_distribution + 1e-6))

        if feedback > 0:
            info_gain = self.last_state_entropy - user_positive_cluster_entropy 
        else:
            info_gain = self.last_state_entropy - user_negative_cluster_entropy 

        self.user_positive_cluster_distribution = user_positive_cluster_distribution
        self.user_negative_cluster_distribution = user_negative_cluster_distribution
        
        return pos_users, pos_nb, neg_users, neg_nb, user_positive_cluster_entropy, user_negative_cluster_entropy, info_gain

    def reset_env(self):
        self.env = self.rating_matrix
        ban_actions = np.ones([self.item_nb,])
        # 修改点：使用 self.cfg.HISTORY_SIZE
        new_pos_state = [0] * self.cfg.HISTORY_SIZE
        new_neg_state = [0] * self.cfg.HISTORY_SIZE
        self.user_index = np.arange(0, self.user_nb)
        self.last_state_entropy = self.init_cluster_entropy
        self.last_distribution = self.init_cluster_p 

        return new_pos_state, new_neg_state, ban_actions