# multi-policy-reinforcement-learning-based-recommender-system

It is the implementation for the paper, A Deep Reinforcement Learning Recommender system with Multiple Policies for Recommendations.

conda activate drl_recommender

/home/chenhongyu/exp/test/
├── data/
│   └── 1m_raw                          # (保持不变) 原始数据文件
│
└── src/                                # 源代码根目录
    ├── main.py                         # (入口) 只负责参数解析和启动训练
    ├── __init__.py
    │
    ├── common/                         # 通用配置和工具
    │   ├── __init__.py
    │   ├── config.py                   # (原 config.py) 存放超参数
    │   └── logger.py                   # (原 logger_utils.py) 日志记录工具
    │
    ├── data_loader/                    # 数据处理模块
    │   ├── __init__.py
    │   ├── dataset.py                  # (原 data_utils.py) 数据加载与预处理
    │   └── replay_buffer.py            # (原 memory.py) 经验回放池
    │
    ├── envs/                           # 环境模块 (解耦出的部分)
    │   ├── __init__.py
    │   └── user_simulator.py           # (从 main.py 提取) User_Enviroment 类
    │
    ├── models/                         # 网络模型模块 (拆分原 model.py)
    │   ├── __init__.py
    │   ├── bpr.py                      # (从 model.py 提取) BPR_MF 类
    │   └── dqn.py                      # (从 model.py 提取) DQNU 类
    │
    └── trainers/                       # 训练逻辑模块 (拆分原 main.py 的逻辑)
        ├── __init__.py
        ├── bpr_trainer.py              # (从 main.py 提取) train_bpr 和聚类逻辑
        └── rl_trainer.py               # (从 main.py 提取) 核心 RL 训练循环




        /home/chenhongyu/exp/test/
├── data/
│   └── 1m_raw                          # (保持不变)
│
└── src/                                # 源代码根目录
    ├── main.py                         # (入口) 这里的 main.py 我们稍后会把逻辑拆分出去
    ├── __init__.py
    │
    ├── common/                         # 放通用工具
    │   ├── __init__.py
    │   ├── config.py                   # ★ 保留原名
    │   └── logger_utils.py             # ★ 保留原名
    │
    ├── data_loader/                    # 放数据处理
    │   ├── __init__.py
    │   ├── data_utils.py               # ★ 保留原名
    │   └── memory.py                   # ★ 保留原名
    │
    ├── envs/                           # 环境 (新拆分出来的)
    │   ├── __init__.py
    │   └── user_env.py                 # (从 main.py 里的 User_Enviroment 类提取)
    │
    ├── models/                         # 模型 (新拆分出来的)
    │   ├── __init__.py
    │   ├── bpr.py                      # (从 model.py 里的 BPR_MF 类提取)
    │   └── dqn.py                      # (从 model.py 里的 DQNU 类提取)
    │
    └── trainers/                       # 训练逻辑 (从 main.py 提取)
        ├── __init__.py
        ├── bpr_trainer.py              # (负责 BPR 预训练)
        └── rl_trainer.py               # (负责 RL 循环)