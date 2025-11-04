import warnings
import torch

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", category=FutureWarning)

# 基础地域关键词库
REGION_KEYWORDS = {
    "北京": ["北京市", "北京"], "上海": ["上海市", "上海"], "广东": ["广东省", "广东"], "四川": ["四川省", "四川"],
    "江苏": ["江苏省", "江苏"], "浙江": ["浙江省", "浙江"], "山东": ["山东省", "山东"], "河南": ["河南省", "河南"],
    "湖北": ["湖北省", "湖北"], "湖南": ["湖南省", "湖南"], "福建": ["福建省", "福建"], "云南": ["云南省", "云南"],
    "陕西": ["陕西省", "陕西"], "贵州": ["贵州省", "贵州"], "广西": ["广西"], "安徽": ["安徽省", "安徽"]
}

# 模型路径配置
MODEL_PATHS = {
    "ner_model": "/data2/jrjiang/realname/model/bert-base-chinese-ner",
    "sentence_model": "/data2/jrjiang/realname/model/bge-large-zh-v1.5"
}

# 聚类参数
CLUSTERING_CONFIG = {
    "min_cluster_size": 3,
    "min_samples": 2,
    "metric": "euclidean",
    "cluster_selection_epsilon": 0.03,
    "alpha": 1.0,
    "cluster_selection_method": "leaf"
}

# 特征融合参数
FEATURE_CONFIG = {
    "entity_weight": 0.15,
    "similarity_threshold": 0.82
}