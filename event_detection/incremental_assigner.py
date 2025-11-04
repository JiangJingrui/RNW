import numpy as np
import torch
from sentence_transformers import util
from config import FEATURE_CONFIG
from feature_extractor import create_entity_vector, combine_features
from event_processor import detect_region_from_text
from utils import clean_text

def assign_new_data_to_events(
    df_new,
    model,
    combined_embeddings_train,
    events,
    ner_extractor,
    entity_vocab,
    similarity_threshold=None):
    
    if similarity_threshold is None:
        similarity_threshold = FEATURE_CONFIG["similarity_threshold"]
    
    print("\n=== 新数据分配到已有事件 ===")
    
    texts_new = df_new["content"].tolist()
    print("编码新数据文本...")
    text_embeddings_new = model.encode(texts_new, convert_to_numpy=True, normalize_embeddings=True)
    
    print("提取新数据实体特征...")
    entity_vectors_new = create_entity_vector(texts_new, ner_extractor, entity_vocab)
    
    combined_new = combine_features(text_embeddings_new, entity_vectors_new, 
                                  entity_weight=FEATURE_CONFIG["entity_weight"])

    # 获取所有已知事件的代表帖 embedding - 使用与训练集相同的特征融合方法
    rep_embeddings = []
    rep_event_ids = []
    
    for event in events:
        # 重新编码代表帖，确保维度一致
        rep_text = event["representative_post_clean"]
        rep_text_emb = model.encode([rep_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # 提取代表帖的实体特征
        rep_entities = ner_extractor.extract_entities(rep_text)
        all_rep_entities = rep_entities['persons'] + rep_entities['locations'] + rep_entities['organizations']
        rep_entity_vector = np.zeros(len(entity_vocab))
        
        for entity in all_rep_entities:
            if entity in entity_vocab:
                idx = entity_vocab.index(entity)
                rep_entity_vector[idx] += 1.0
        
        # 融合特征（与训练集相同的方法）
        rep_combined = combine_features(
            rep_text_emb.reshape(1, -1), 
            rep_entity_vector.reshape(1, -1), 
            entity_weight=FEATURE_CONFIG["entity_weight"]
        )[0]
        
        rep_embeddings.append(rep_combined)
        rep_event_ids.append(event["event_id"])
    
    # 转换为tensor并确保在CPU上
    rep_embeddings = np.array(rep_embeddings)
    rep_embs = torch.tensor(rep_embeddings, dtype=torch.float32)
    
    # 归一化
    rep_embs = rep_embs / (torch.norm(rep_embs, dim=1, keepdim=True) + 1e-8)
    
    assignments = []
    new_event_count = 0
    
    print("开始分配新数据到事件...")
    for i, emb in enumerate(combined_new):
        if i % 100 == 0:
            print(f"  分配进度: {i}/{len(combined_new)}")
            
        # 确保新数据embedding与代表帖embedding维度一致
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        emb_tensor = emb_tensor / (torch.norm(emb_tensor, dim=1, keepdim=True) + 1e-8)
        
        # 计算相似度
        sims = torch.mm(emb_tensor, rep_embs.T).numpy()[0]
        
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        
        if best_sim >= similarity_threshold:
            assigned_event = rep_event_ids[best_idx]
            for event in events:
                if event["event_id"] == assigned_event:
                    event["all_posts"].append({
                        "id": df_new.iloc[i]["id"] if "id" in df_new.columns else f"new_{i}",
                        "text": df_new.iloc[i]["text"]
                    })
                    event["post_count"] = len(event["all_posts"])
                    break
        else:
            new_event_count += 1
            assigned_event = f"new_{new_event_count}"
            events.append({
                "event_id": assigned_event,
                "representative_post": df_new.iloc[i]["text"],
                "representative_post_clean": clean_text(df_new.iloc[i]["text"]),
                "representative_similarity": 1.0,
                "all_posts": [{
                    "id": df_new.iloc[i]["id"] if "id" in df_new.columns else f"new_{i}",
                    "text": df_new.iloc[i]["text"]
                }],
                "post_count": 1,
                "time_span_analysis": {},
                "interaction_statistics": {},
                "region": detect_region_from_text(df_new.iloc[i]["text"], ner_extractor)
            })
        
        assignments.append((i, assigned_event, best_sim))
    
    matched_count = len([a for a in assignments if 'new_' not in a[1]])
    print(f"新数据分配完成：已有事件匹配 {matched_count} 条，新增事件 {new_event_count} 个")
    return events, assignments