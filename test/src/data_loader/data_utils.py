import random, os
import time
import numpy as np
import pickle as pkl

class Utils(object):
    def __init__(self):
        pass

    def load_raw_data_rating(self, path):
        # 增加容错：如果路径是目录，自动寻找 ratings.dat
        if os.path.isdir(path):
            file_path = os.path.join(path, 'ratings.dat')
            if not os.path.exists(file_path):
                # 如果没有 ratings.dat，尝试找目录下的第一个文件
                files = [f for f in os.listdir(path) if not f.startswith('.')]
                if files:
                    file_path = os.path.join(path, files[0])
                else:
                    raise FileNotFoundError(f"No data file found in {path}")
            print(f"    Target is a directory, automatically reading: {file_path}")
            f = open(file_path, 'r')
        else:
            f = open(path, 'r')

        user_historicals = {}
        user_dict = {}
        item_dict = {}
        interactive_length = {}
        user_count = 0
        item_count = 1  # 这里的 item_count 将作为 item_id 的上限 + 1

        raw_item_dict = {}

        for line in f:
            # 兼容 :: 分隔符 (MovieLens) 或 \t 分隔符
            if '::' in line:
                data = line.split('::')
            else:
                data = line.split('\t')
                
            user = int(data[0])
            item = int(data[1])
            rating = float(data[2])
            time_stmp = int(data[3].strip())

            if user not in user_dict.keys():
                user_dict[user] = user_count
                user_count += 1

            if item not in item_dict.keys():
                item_dict[item] = item_count
                item_count += 1

            user = user_dict[user]

            raw_item_dict[item] = item_dict[item]
            item = item_dict[item]

            if user not in user_historicals.keys():
                user_historicals[user] = []
            user_historicals[user].append((user, item, rating, time_stmp))

        f.close()

        user_rating_matrix = {}

        for user in user_historicals.keys():
            user_historicals[user] = sorted(user_historicals[user], key=lambda a: a[-1])

            interactive_length[user] = len(user_historicals[user])

            # 注意：这里的 shape 必须能容纳最大的 item index
            user_rating_vector = np.zeros(shape=[item_count], dtype=np.uint8)

            for i in user_historicals[user]:
                user_rating_vector[i[1]] = i[2]

            user_historicals[user] = [d[1] for d in user_historicals[user]]
            user_rating_matrix[user] = user_rating_vector.copy()

        # 【重要】多返回一个 item_count 以修正 Config
        return user_historicals, user_rating_matrix, interactive_length, item_count