import pandas as pd
import numpy as np
import hdbscan
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

from config import MODEL_PATHS, CLUSTERING_CONFIG, FEATURE_CONFIG
from utils import clean_text, convert_to_serializable, save_events_to_json
from feature_extractor import NERFeatureExtractor, build_entity_vocab, create_entity_vector, combine_features
from event_processor import select_robust_representative, detect_region_from_text, recompute_representative_for_event
from post_processor import post_process_clusters
from incremental_assigner import assign_new_data_to_events

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    df_full = pd.read_csv(file_path)
    if "发布时间" in df_full.columns:
        df_full["发布时间"] = pd.to_datetime(df_full["发布时间"], errors="coerce")
    else:
        df_full["发布时间"] = pd.NaT
    
    df_full["content"] = df_full["text"].astype(str).apply(clean_text)
    print(f"读取数据：总行数 {len(df_full)}")
    return df_full

def build_initial_events(train_df, combined_embeddings, texts_all_train, ner_extractor):
    """构建初始事件结构"""
    interaction_columns = ['like_count', 'favorite_count', 'comment_count', 'share_count', 'tags']
    available_columns = [col for col in interaction_columns if col in train_df.columns]
    print(f"可用的互动统计列: {available_columns}")
    
    events = []
    for eid, group in train_df.groupby("event_id"):
        group_indices = group.index.tolist()
        print(f"\n处理事件 {eid}，包含 {len(group_indices)} 个帖子")
        
        rep_index, rep_text, rep_similarity = select_robust_representative(
            group_indices, combined_embeddings, texts_all_train, train_df
        )
        
        print(f"事件 {eid}: 选择代表帖，综合得分 {rep_similarity:.4f}")
        
        all_posts = []
        for _, row in group.iterrows():
            post_data = {
                "id": convert_to_serializable(row["id"]) if "id" in train_df.columns else None,
                "text": str(row["text"])
            }
            if 'like_count' in train_df.columns:
                post_data["like_count"] = convert_to_serializable(row["like_count"])
            if 'favorite_count' in train_df.columns:
                post_data["favorite_count"] = convert_to_serializable(row["favorite_count"])
            if 'comment_count' in train_df.columns:
                post_data["comment_count"] = convert_to_serializable(row["comment_count"])
            if 'share_count' in train_df.columns:
                post_data["share_count"] = convert_to_serializable(row["share_count"])
            if 'tags' in train_df.columns:
                post_data["tags"] = convert_to_serializable(row["tags"])
            if "发布时间" in train_df.columns:
                post_data["发布时间"] = convert_to_serializable(row["发布时间"])
            all_posts.append(post_data)
        
        # 交互统计
        interaction_statistics = {}
        if 'like_count' in train_df.columns:
            interaction_statistics["total_likes"] = convert_to_serializable(group['like_count'].sum())
            interaction_statistics["avg_likes"] = convert_to_serializable(round(group['like_count'].mean(), 2))
        if 'favorite_count' in train_df.columns:
            interaction_statistics["total_favorites"] = convert_to_serializable(group['favorite_count'].sum())
            interaction_statistics["avg_favorites"] = convert_to_serializable(round(group['favorite_count'].mean(), 2))
        if 'comment_count' in train_df.columns:
            interaction_statistics["total_comments"] = convert_to_serializable(group['comment_count'].sum())
            interaction_statistics["avg_comments"] = convert_to_serializable(round(group['comment_count'].mean(), 2))
        if 'share_count' in train_df.columns:
            interaction_statistics["total_shares"] = convert_to_serializable(group['share_count'].sum())
            interaction_statistics["avg_shares"] = convert_to_serializable(round(group['share_count'].mean(), 2))
        
        # 时间跨度
        time_span_analysis = {}
        if "发布时间" in group.columns and not group["发布时间"].isna().all():
            valid_times = group["发布时间"].dropna()
            if len(valid_times) > 0:
                start, end = valid_times.min(), valid_times.max()
                duration_days = (end - start).total_seconds() / (24 * 3600)
                avg_posts_per_day = len(valid_times) / max(duration_days, 1)
                
                time_span_analysis = {
                    "start_time": convert_to_serializable(start),
                    "end_time": convert_to_serializable(end),
                    "duration_days": convert_to_serializable(round(duration_days, 2)),
                    "post_count_temporal": convert_to_serializable(len(valid_times)),
                    "avg_posts_per_day": convert_to_serializable(round(avg_posts_per_day, 2))
                }
            else:
                time_span_analysis = {"message": "该簇中没有有效的时间数据"}
        else:
            time_span_analysis = {"message": "数据集中没有时间列"}
        
        # 地域检测
        combined_text = " ".join([post["text"] for post in all_posts[:3]])
        region = detect_region_from_text(combined_text, ner_extractor)
        
        events.append({
            "event_id": str(eid),
            "representative_post": rep_text,
            "representative_post_clean": clean_text(rep_text),
            "representative_similarity": float(rep_similarity),
            "all_posts": all_posts,
            "post_count": len(group),
            "time_span_analysis": time_span_analysis,
            "interaction_statistics": interaction_statistics,
            "region": region
        })
    
    return events, available_columns

def main():
    # 1. 读取与清洗
    raw_path = "/data2/jrjiang/realname/data/use/is_realname_positive_samples.csv"
    df_full = load_and_preprocess_data(raw_path)
    
    # 随机拆分
    train_df, new_df = train_test_split(df_full, test_size=0.3, random_state=42)
    train_df = train_df.reset_index(drop=True)
    new_df = new_df.reset_index(drop=True)
    print(f"训练集: {len(train_df)}，新增数据: {len(new_df)}")
    
    # 2. 初始化 NER / 模型
    ner_extractor = NERFeatureExtractor()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_PATHS["sentence_model"])
    
    # 3. 构建实体词汇表
    train_texts = train_df["content"].astype(str).tolist()
    entity_vocab = build_entity_vocab(train_texts, ner_extractor)
    
    # 4. 为训练集创建特征
    print("为训练集创建实体特征向量...")
    entity_vectors_train = create_entity_vector(train_texts, ner_extractor, entity_vocab) if len(entity_vocab) > 0 else np.zeros((len(train_texts), 0))
    
    print("开始训练集文本向量化...")
    text_embeddings_train = model.encode(train_texts, 
                                 convert_to_numpy=True, 
                                 normalize_embeddings=True,
                                 batch_size=32,
                                 show_progress_bar=True)
    print(f"训练集文本向量化完成，维度: {text_embeddings_train.shape}")
    print(f"实体特征维度: {entity_vectors_train.shape}")
    
    print("融合文本和实体特征（训练集）...")
    combined_embeddings_train = combine_features(text_embeddings_train, entity_vectors_train, 
                                               entity_weight=FEATURE_CONFIG["entity_weight"])
    print(f"训练集融合后维度: {combined_embeddings_train.shape}")
    
    # 5. HDBSCAN 聚类
    print("开始训练集聚类（HDBSCAN）...")
    clusterer = hdbscan.HDBSCAN(**CLUSTERING_CONFIG)
    labels = clusterer.fit_predict(combined_embeddings_train)
    train_df["event_id"] = labels
    next_event_id = int(train_df["event_id"].max()) + 1 if len(train_df) > 0 else 0
    for idx in train_df[train_df["event_id"] == -1].index:
        train_df.at[idx, "event_id"] = next_event_id
        next_event_id += 1
    print(f"训练集聚类完成，共 {train_df['event_id'].nunique()} 个事件簇（训练集）")
    
    # 6. 构建初始 events 结构
    events, available_columns = build_initial_events(train_df, combined_embeddings_train, train_texts, ner_extractor)
    print(f"初步事件数: {len(events)}")
    
    # 7. 聚类后处理
    events = post_process_clusters(events, train_df.assign(event_id=train_df["event_id"]), 
                                 combined_embeddings_train, train_texts, ner_extractor)
    
    # 保存初始结果
    output_dir = "/data2/jrjiang/realname/data"
    os.makedirs(output_dir, exist_ok=True)
    initial_output_path = os.path.join(output_dir, "events_result_robust.json")
    
    initial_metadata = {
        "clustering_method": "HDBSCAN with robust representative selection",
        "feature_fusion": f"text_embeddings + entity_vectors (entity_weight={FEATURE_CONFIG['entity_weight']})",
        "total_events": len(events),
        "total_posts": len(train_df),
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "available_interaction_columns": available_columns,
        "entity_vocab_size": len(entity_vocab),
        "post_processing": "enabled"
    }
    
    save_events_to_json(events, initial_output_path, initial_metadata)
    
    # 8. 增量分配
    new_df_local = new_df.copy().reset_index(drop=True)
    new_df_local["content"] = new_df_local["text"].astype(str).apply(clean_text)
    new_df_local["发布时间"] = pd.to_datetime(new_df_local["发布时间"], errors="coerce") if "发布时间" in new_df_local.columns else pd.NaT
    
    events_after_assign, assignments = assign_new_data_to_events(
        new_df_local,
        model,
        combined_embeddings_train,
        events,
        ner_extractor,
        entity_vocab
    )
    
    # 9. 重新计算统计信息
    print("为所有事件重新计算代表帖与统计...")
    final_events = []
    for ev in events_after_assign:
        rep_text, rep_sim = recompute_representative_for_event(ev, model)
        ev["representative_post"] = rep_text
        ev["representative_post_clean"] = clean_text(rep_text)
        ev["representative_similarity"] = rep_sim
        
        # 重新计算交互统计
        interaction_statistics = {}
        likes = [p.get("like_count", 0) for p in ev.get("all_posts", []) if p.get("like_count") is not None]
        if len(likes) > 0:
            total_likes = sum(likes)
            interaction_statistics["total_likes"] = convert_to_serializable(total_likes)
            interaction_statistics["avg_likes"] = convert_to_serializable(round(total_likes / max(len(likes),1), 2))
        favs = [p.get("favorite_count", 0) for p in ev.get("all_posts", []) if p.get("favorite_count") is not None]
        if len(favs) > 0:
            total_favs = sum(favs)
            interaction_statistics["total_favorites"] = convert_to_serializable(total_favs)
            interaction_statistics["avg_favorites"] = convert_to_serializable(round(total_favs / max(len(favs),1), 2))
        cmts = [p.get("comment_count", 0) for p in ev.get("all_posts", []) if p.get("comment_count") is not None]
        if len(cmts) > 0:
            total_cmts = sum(cmts)
            interaction_statistics["total_comments"] = convert_to_serializable(total_cmts)
            interaction_statistics["avg_comments"] = convert_to_serializable(round(total_cmts / max(len(cmts),1), 2))
        shares = [p.get("share_count", 0) for p in ev.get("all_posts", []) if p.get("share_count") is not None]
        if len(shares) > 0:
            total_shares = sum(shares)
            interaction_statistics["total_shares"] = convert_to_serializable(total_shares)
            interaction_statistics["avg_shares"] = convert_to_serializable(round(total_shares / max(len(shares),1), 2))
        ev["interaction_statistics"] = interaction_statistics
        
        # 重新计算时间跨度
        times = []
        for p in ev.get("all_posts", []):
            t = p.get("发布时间", None)
            if t:
                try:
                    times.append(pd.to_datetime(t))
                except:
                    pass
        if times:
            start, end = min(times), max(times)
            duration_days = (end - start).total_seconds() / (24 * 3600)
            ev["time_span_analysis"] = {
                "start_time": convert_to_serializable(start),
                "end_time": convert_to_serializable(end),
                "duration_days": convert_to_serializable(round(duration_days, 2)),
                "post_count_temporal": convert_to_serializable(len(times)),
                "avg_posts_per_day": convert_to_serializable(round(len(times) / max(duration_days, 1), 2))
            }
        else:
            ev["time_span_analysis"] = {"message": "合并后无有效时间数据"}
        
        # 重新检测地域
        if not ev.get("region"):
            combined_text = " ".join([p.get("text","") for p in ev.get("all_posts", [])[:5]])
            ev["region"] = detect_region_from_text(combined_text, ner_extractor)
        
        ev["post_count"] = len(ev.get("all_posts", []))
        final_events.append(ev)
    
    # 10. 最终后处理
    final_events = post_process_clusters(final_events, pd.concat([train_df, new_df_local.assign(event_id=None)], ignore_index=True), 
                                       combined_embeddings_train, train_texts, ner_extractor)
    
    # 11. 保存最终结果
    output_path = os.path.join(output_dir, "events_result_final_robust.json")
    final_metadata = {
        "clustering_method": "HDBSCAN with robust representative selection",
        "feature_fusion": f"text_embeddings + entity_vectors (entity_weight={FEATURE_CONFIG['entity_weight']})",
        "total_events": len(final_events),
        "total_posts": len(df_full),
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "available_interaction_columns": available_columns,
        "entity_vocab_size": len(entity_vocab),
        "post_processing": "enabled",
        "train_size": len(train_df),
        "new_data_size": len(new_df)
    }
    
    save_events_to_json(final_events, output_path, final_metadata)
    
    # 12. 打印摘要
    print_summary(final_events, available_columns, entity_vocab)

def print_summary(events, available_columns, entity_vocab):
    """打印分析摘要"""
    print("\n=== 聚类分析摘要 ===")
    print(f"最终事件簇数: {len(events)}")
    print(f"使用的特征: 文本向量 + 实体向量 (实体词汇表大小: {len(entity_vocab)})")
    print(f"可用的互动统计列: {available_columns}")
    
    region_stats = {}
    for e in events:
        region = e.get('region', '未知')
        region_stats[region] = region_stats.get(region, 0) + 1
    
    print(f"\n地域分布: {region_stats}")
    
    print("\n前10个事件：")
    for e in events[:10]:
        region_info = f" [{e.get('region', '未知')}]" if 'region' in e else ""
        print(f"- 事件 {e['event_id']}{region_info} ({e['post_count']} 帖): {e['representative_post_clean'][:80]}...")
        print(f"  代表帖质量: {e.get('representative_similarity', 0):.4f}")
        if e.get('interaction_statistics'):
            stats = []
            if 'total_likes' in e['interaction_statistics']:
                stats.append(f"点赞:{e['interaction_statistics']['total_likes']}")
            if 'total_comments' in e['interaction_statistics']:
                stats.append(f"评论:{e['interaction_statistics']['total_comments']}")
            if 'total_shares' in e['interaction_statistics']:
                stats.append(f"分享:{e['interaction_statistics']['total_shares']}")
            if stats:
                print(f"  互动统计: {', '.join(stats)}")

if __name__ == "__main__":
    main()