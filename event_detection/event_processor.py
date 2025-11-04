import numpy as np
import pandas as pd
from sentence_transformers import util
import torch
from config import REGION_KEYWORDS
from utils import clean_text, convert_to_serializable

def select_robust_representative(group_indices, combined_embeddings, texts, df=None):
    """鲁棒的代表帖选择：考虑多种因素"""
    if len(group_indices) == 0:
        return None, "", 0.0
    
    if len(group_indices) == 1:
        idx = group_indices[0]
        return idx, texts[idx], 1.0
    
    cluster_embeddings = combined_embeddings[group_indices]
    
    # 方法1: 基于平均相似度
    similarity_matrix = util.cos_sim(cluster_embeddings, cluster_embeddings).cpu().numpy()
    avg_similarities = []
    for i in range(len(group_indices)):
        other_similarities = np.delete(similarity_matrix[i], i)
        avg_sim = np.mean(other_similarities) if len(other_similarities) > 0 else 0.0
        avg_similarities.append(avg_sim)
    
    # 方法2: 基于中心点距离
    cluster_center = np.mean(cluster_embeddings, axis=0)
    center_distances = []
    for i in range(len(group_indices)):
        distance = 1 - util.cos_sim(torch.tensor(cluster_embeddings[i]), torch.tensor(cluster_center)).item()
        center_distances.append(distance)
    
    # 方法3: 基于文本质量（长度和互动）
    text_qualities = []
    for i, idx in enumerate(group_indices):
        text = texts[idx]
        # 文本长度得分 (50-500字符为佳)
        length_score = min(1.0, max(0.0, (len(text) - 50) / 450))
        
        # 互动得分（如果有数据）
        interaction_score = 0.5  # 默认值
        if df is not None and 'like_count' in df.columns:
            try:
                likes = df.iloc[idx].get('like_count', 0)
                # 对数尺度，避免极端值影响
                interaction_score = min(1.0, np.log1p(likes) / 10)
            except:
                interaction_score = 0.5
        
        quality_score = 0.6 * length_score + 0.4 * interaction_score
        text_qualities.append(quality_score)
    
    # 综合评分
    final_scores = []
    for i in range(len(group_indices)):
        # 权重：相似度50%，中心距离30%，文本质量20%
        score = (0.5 * avg_similarities[i] + 
                 0.3 * (1 - center_distances[i]) + 
                 0.2 * text_qualities[i])
        final_scores.append(score)
    
    best_idx_in_cluster = np.argmax(final_scores)
    representative_index = group_indices[best_idx_in_cluster]
    representative_text = texts[representative_index]
    best_score = final_scores[best_idx_in_cluster]
    
    print(f"  选择代表帖: 索引={representative_index}, 综合得分={best_score:.4f}")
    print(f"  代表帖内容: {representative_text[:100]}...")
    
    return representative_index, representative_text, best_score

def detect_region_from_text(text, ner_extractor):
    """从文本中检测地域信息"""
    for region, keywords in REGION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return region
    
    entities = ner_extractor.extract_entities(text)
    for location in entities['locations']:
        for region, keywords in REGION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in location:
                    return region
    
    return "未知"

def recompute_representative_for_event(event, model):
    """重新计算事件代表帖"""
    texts = [p.get("text", "") for p in event.get("all_posts", [])]
    if len(texts) == 0:
        return "", 0.0
    if len(texts) == 1:
        return texts[0], 1.0
    
    valid_texts = [text for text in texts if text and len(text.strip()) > 0]
    if len(valid_texts) == 0:
        return texts[0] if texts else "", 0.0
    if len(valid_texts) == 1:
        return valid_texts[0], 1.0
        
    embs = model.encode(valid_texts, convert_to_tensor=True, normalize_embeddings=True)
    sim_mat = util.cos_sim(embs, embs).cpu().numpy()
    avg_sims = sim_mat.mean(axis=1)
    best_idx = int(np.argmax(avg_sims))
    return valid_texts[best_idx], float(avg_sims[best_idx])