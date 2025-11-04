import re
import pandas as pd
import numpy as np
from datetime import datetime
import json

def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""
    text = re.sub(r"#.*?#", "", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def convert_to_serializable(obj):
    """转换为可序列化的Python类型"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(obj) else None
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif pd.isna(obj):
        return None
    return obj

def save_events_to_json(events, output_path, metadata=None):
    """保存事件结果到JSON文件"""
    output_data = {
        "metadata": metadata or {},
        "events": events
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {output_path}")