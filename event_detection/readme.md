🏗️ 系统架构
├── config.py              # 配置参数和常量定义
├── utils.py               # 通用工具函数
├── feature_extractor.py   # 特征提取模块（NER、文本编码）
├── event_processor.py     # 事件处理逻辑
├── post_processor.py      # 聚类后处理
├── incremental_assigner.py # 增量数据分配
├── main.py               # 主程序入口
└── README.md             # 项目说明文档


模型路径配置
在 config.py 中修改：

python
MODEL_PATHS = {
    "ner_model": "/path/to/your/ner/model",
    "sentence_model": "/path/to/your/sentence/model"
}

聚类参数调整
python
CLUSTERING_CONFIG = {
    "min_cluster_size": 3,      # 最小簇大小
    "min_samples": 2,           # 核心点最小样本数
    "metric": "euclidean",      # 距离度量
    "cluster_selection_epsilon": 0.03,  # 簇选择阈值
    "alpha": 1.0,               # 距离计算参数
    "cluster_selection_method": "leaf"  # 簇选择方法
}

特征融合参数
python
FEATURE_CONFIG = {
    "entity_weight": 0.15,           # 实体特征权重
    "similarity_threshold": 0.82     # 相似度阈值
}

📊 输出结果
系统生成两个主要输出文件：

1. 初始聚类结果
events_result_robust.json - 训练数据的聚类结果

2. 最终结果
events_result_final_robust.json - 包含增量数据的完整结果


新增report_analysis.py可用于结果的分析