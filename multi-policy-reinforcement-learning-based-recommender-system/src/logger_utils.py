import os
import csv
import tensorflow as tf
from datetime import datetime

class TrainingLogger:
    def __init__(self, config):
        self.config = config
        # 创建以时间戳命名的文件夹，防止多次实验数据覆盖
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"../experiments/{config.DATA_SET}_{self.timestamp}"
        self.model_dir = os.path.join(self.base_dir, "checkpoints")
        self.log_file = os.path.join(self.base_dir, "train_metrics.csv")
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 初始化 CSV 文件头
        self._init_csv()
        
        # 初始化 TF 保存器 (针对 TF 1.x)
        self.saver = None

    def _init_csv(self):
        headers = ['episode', 'avg_test_hit', 'phase1_hit', 'phase2_hit', 'avg_ndcg', 'train_loss']
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_metrics(self, episode, hit, p1_hit, p2_hit, ndcg, loss):
        """记录指标到 CSV"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, f"{hit:.4f}", f"{p1_hit:.4f}", f"{p2_hit:.4f}", f"{ndcg:.4f}", f"{loss:.4f}"])
        print(f"--- Metrics saved to {self.log_file} ---")

    def save_model(self, sess, step):
        """保存模型权重"""
        if self.saver is None:
            # 延迟初始化，确保所有变量已在图中创建
            self.saver = tf.train.Saver(max_to_keep=5)
        
        save_path = os.path.join(self.model_dir, "model.ckpt")
        self.saver.save(sess, save_path, global_step=step)
        print(f"--- Model checkpoint saved at step {step} ---")