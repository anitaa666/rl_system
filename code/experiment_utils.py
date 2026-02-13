# -*- coding: utf-8 -*-
import os
import tensorflow as tf

def init_experiment(model_type, test_num):
    """
    初始化实验路径
    :param model_type: 'cold' 或 'warm'
    :param test_num: 实验组编号 (如 '0', '1')
    :return: (model_dir, log_file)
    """
    base_dir = "./experiments/%s_test_%s" % (model_type, test_num)
    model_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    
    # 创建目录
    for d in [model_dir, log_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    log_file = os.path.join(log_dir, "results.txt")
    return model_dir, log_file

def save_log(log_file, epoch, train_hr, test_hr, test_ndcg, diversity):
    """记录实验指标"""
    with open(log_file, 'a') as f:
        line = "Epoch: %d | Train_HR: %.4f | Test_HR: %.4f | NDCG@10: %.4f | Diversity: %.4f\n" % \
               (epoch, train_hr, test_hr, test_ndcg, diversity)
        f.write(line)
    # 同时在控制台显示，方便观察
    print(">>> Log saved to %s" % log_file)

def save_model(saver, sess, model_dir, epoch):
    """保存模型权重"""
    save_path = os.path.join(model_dir, "model.ckpt")
    saver.save(sess, save_path, global_step=epoch)
    print(">>> Model checkpoint saved at: %s-%d" % (save_path, epoch))