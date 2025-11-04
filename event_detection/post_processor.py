import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from config import MODEL_PATHS, FEATURE_CONFIG
from event_processor import detect_region_from_text, select_robust_representative
from utils import clean_text, convert_to_serializable

def post_process_clusters(events, df, combined_embeddings, texts, ner_extractor, similarity_threshold=None):
    """后处理聚类结果：合并过度分割的簇"""
    if similarity_threshold is None:
        similarity_threshold = FEATURE_CONFIG["similarity_threshold"]
    
    if len(events) <= 1:
        return events
    
    print("进行聚类后处理，合并过度分割的簇...")
    
    rep_texts = [e["representative_post_clean"] for e in events]
    rep_embs = SentenceTransformer(MODEL_PATHS["sentence_model"]).encode(
        rep_texts, convert_to_tensor=True, normalize_embeddings=True
    )
    
    cos_sim = util.cos_sim(rep_embs, rep_embs).cpu().numpy()
    
    # 检测每个事件的地域
    event_regions = []
    for event in events:
        combined_text = " ".join([post.get("text", "") for post in event.get("all_posts", [])[:3]])
        region = detect_region_from_text(combined_text, ner_extractor)
        event_regions.append(region)
    
    merged = []
    visited = set()
    
    for i in range(len(events)):
        if i in visited:
            continue
        
        similar_idxs = {i}
        current_region = event_regions[i]
        
        for j in range(i + 1, len(events)):
            if (cos_sim[i][j] >= similarity_threshold and 
                event_regions[j] == current_region and
                current_region != "未知"):
                similar_idxs.add(j)
            elif cos_sim[i][j] >= 0.85:  # 高相似度时忽略地域
                similar_idxs.add(j)
        
        visited.update(similar_idxs)
        
        if len(similar_idxs) == 1:
            merged.append(events[i])
            continue
        
        print(f"合并事件: {[events[idx]['event_id'] for idx in similar_idxs]}")
        
        merged_posts = []
        total_likes = 0
        total_favorites = 0
        total_comments = 0
        total_shares = 0
        
        all_start_times = []
        all_end_times = []
        all_indices = []
        
        for idx in similar_idxs:
            event = events[idx]
            merged_posts.extend(event.get("all_posts", []))
            
            interaction_stats = event.get("interaction_statistics", {})
            total_likes += interaction_stats.get("total_likes", 0)
            total_favorites += interaction_stats.get("total_favorites", 0)
            total_comments += interaction_stats.get("total_comments", 0)
            total_shares += interaction_stats.get("total_shares", 0)
            
            time_span = event.get("time_span_analysis", {})
            if time_span and "start_time" in time_span and time_span["start_time"]:
                try:
                    start_time = pd.to_datetime(time_span["start_time"])
                    all_start_times.append(start_time)
                except:
                    pass
            if time_span and "end_time" in time_span and time_span["end_time"]:
                try:
                    end_time = pd.to_datetime(time_span["end_time"])
                    all_end_times.append(end_time)
                except:
                    pass
            
            group_for_event = df[df["event_id"] == events[idx]["event_id"]] if "event_id" in df.columns else pd.DataFrame()
            all_indices.extend(group_for_event.index.tolist())
        
        if all_indices:
            rep_index, rep_text, rep_similarity = select_robust_representative(
                all_indices, combined_embeddings, texts, df
            )
        else:
            rep_text = merged_posts[0]["text"] if merged_posts else ""
            rep_similarity = 1.0 if merged_posts else 0.0
        
        post_count = len(merged_posts)
        merged_interaction_stats = {}
        if total_likes > 0:
            merged_interaction_stats["total_likes"] = total_likes
            merged_interaction_stats["avg_likes"] = round(total_likes / max(post_count, 1), 2)
        if total_favorites > 0:
            merged_interaction_stats["total_favorites"] = total_favorites
            merged_interaction_stats["avg_favorites"] = round(total_favorites / max(post_count, 1), 2)
        if total_comments > 0:
            merged_interaction_stats["total_comments"] = total_comments
            merged_interaction_stats["avg_comments"] = round(total_comments / max(post_count, 1), 2)
        if total_shares > 0:
            merged_interaction_stats["total_shares"] = total_shares
            merged_interaction_stats["avg_shares"] = round(total_shares / max(post_count, 1), 2)
        
        merged_time_span = {}
        if all_start_times and all_end_times:
            start_time = min(all_start_times)
            end_time = max(all_end_times)
            duration_days = (end_time - start_time).total_seconds() / (24 * 3600)
            
            merged_time_span = {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_days": round(duration_days, 2),
                "post_count_temporal": post_count,
                "avg_posts_per_day": round(post_count / max(duration_days, 1), 2)
            }
        else:
            merged_time_span = {"message": "合并后无有效时间数据"}
        
        combined_text = " ".join([post["text"] for post in merged_posts[:5]])
        region = detect_region_from_text(combined_text, ner_extractor)
        
        merged_event = {
            "event_id": f"merged_{len(merged) + 1}",
            "representative_post": rep_text,
            "representative_post_clean": clean_text(rep_text),
            "representative_similarity": float(rep_similarity),
            "all_posts": merged_posts,
            "post_count": post_count,
            "time_span_analysis": merged_time_span,
            "interaction_statistics": merged_interaction_stats,
            "region": region,
            "merged_from": [events[idx]["event_id"] for idx in similar_idxs],
            "merged_count": len(similar_idxs)
        }
        
        merged.append(merged_event)
    
    print(f"后处理完成: {len(events)} -> {len(merged)} 个事件")
    return merged