
"""
实名举报事件舆论分析（纯文字版）
--------------------------------------
功能：
1. 分析事件热度、互动、地区分布、时间趋势
2. 输出简洁文字报告（控制台打印 + 文件保存）
运行方式：
    python public_opinion_analysis_text.py
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime


# =============================
# 数据加载与基础计算
# =============================

def load_event_data(json_path):
    """加载事件聚类结果"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["events"]

def compute_event_metrics(event):
    """计算单个事件的舆论指标"""
    stats = event.get("interaction_statistics", {})
    region = event.get("region", "未知")
    rep_post = event.get("representative_post_clean", "")[:60]
    duration = event.get("time_span_analysis", {}).get("duration_days", 0)
    tinfo = event.get("time_span_analysis", {})

    total_heat = (
        stats.get("total_likes", 0)
        + 2 * stats.get("total_comments", 0)
        + 3 * stats.get("total_shares", 0)
        + stats.get("total_favorites", 0)
    )
    avg_heat = (
        stats.get("avg_likes", 0)
        + 2 * stats.get("avg_comments", 0)
        + 3 * stats.get("avg_shares", 0)
        + stats.get("avg_favorites", 0)
    )

    return {
        "event_id": event.get("event_id"),
        "region": region,
        "代表帖": rep_post,
        "帖子数": event.get("post_count", 0),
        "平均点赞": stats.get("avg_likes", 0),
        "平均评论": stats.get("avg_comments", 0),
        "平均转发": stats.get("avg_shares", 0),
        "平均收藏": stats.get("avg_favorites", 0),
        "总热度指数": total_heat,
        "平均热度指数": avg_heat,
        "传播时长(天)": duration,
        "开始时间": tinfo.get("start_time"),
        "结束时间": tinfo.get("end_time"),
    }

def analyze_events(events):
    """生成完整分析表"""
    metrics = [compute_event_metrics(e) for e in events]
    df = pd.DataFrame(metrics)
    df = df.sort_values(by="总热度指数", ascending=False).reset_index(drop=True)
    return df


# =============================
# 文本分析与报告生成
# =============================

def summarize_analysis(df):
    """生成文字分析报告"""

    lines = []
    lines.append("📊 实名举报事件舆论分析报告")
    lines.append("=" * 50)

    # 总体概况
    lines.append(f"\n共统计事件数量: {len(df)}")
    lines.append(f"平均帖子数: {df['帖子数'].mean():.1f}")
    lines.append(f"平均热度指数: {df['总热度指数'].mean():.1f}")
    lines.append(f"平均传播时长: {df['传播时长(天)'].mean():.1f} 天")

    # 热度前五事件
    lines.append("\n🔥 热度最高的 5 个事件：")
    for i, row in df.head(5).iterrows():
        lines.append(
            f"{i+1}. {row['代表帖']} | 地区: {row['region']} | 热度指数: {row['总热度指数']:,} | "
            f"平均点赞: {row['平均点赞']:.0f} | 转发: {row['平均转发']:.0f} | 评论: {row['平均评论']:.0f}"
        )

    # 地区分布
    region_counts = df["region"].value_counts()
    lines.append("\n🌍 举报事件地区分布：")
    for region, count in region_counts.items():
        lines.append(f" - {region}: {count} 起 ({count/len(df)*100:.1f}%)")

    # 时间趋势分析
    df["开始时间_dt"] = pd.to_datetime(df["开始时间"], errors="coerce")
    time_df = df.dropna(subset=["开始时间_dt"]).sort_values("开始时间_dt")

    if not time_df.empty:
        first = time_df["开始时间_dt"].iloc[0]
        last = time_df["开始时间_dt"].iloc[-1]
        lines.append(f"\n⏰ 数据时间范围：{first.date()} ~ {last.date()}")
        # 热度随时间变化趋势
        early = time_df.head(len(time_df)//3)["总热度指数"].mean()
        mid = time_df.iloc[len(time_df)//3: 2*len(time_df)//3]["总热度指数"].mean()
        late = time_df.tail(len(time_df)//3)["总热度指数"].mean()
        trend = "上升" if late > early else "下降" if late < early else "持平"
        lines.append(f"总体热度趋势：{trend}（早期均值={early:.0f}, 后期均值={late:.0f}）")

    # 输出摘要
    report = "\n".join(lines)
    return report


# =============================
# 标签统计
# =============================

def analyze_tags(events, top_n=15):
    """统计高频标签"""
    all_tags = []
    for e in events:
        for p in e.get("all_posts", []):
            tags = p.get("tags")
            if not tags:
                continue
            if isinstance(tags, str):
                try:
                    tags_list = eval(tags)
                except Exception:
                    tags_list = [tags]
            else:
                tags_list = tags
            all_tags.extend(tags_list)
    counter = Counter(all_tags)
    return counter.most_common(top_n)


# =============================
# 主程序
# =============================

def main(json_path="/data2/jrjiang/realname/data/events_result_final_robust.json"):
    print("🚀 正在加载数据并生成舆论分析报告...")
    events = load_event_data(json_path)
    df = analyze_events(events)

    # 文字报告
    report = summarize_analysis(df)

    # 高频标签
    tags = analyze_tags(events)
    tag_text = "\n\n#️⃣ 高频标签：\n" + "\n".join([f"{i+1}. {t[0]} ({t[1]} 次)" for i, t in enumerate(tags)])
    report += tag_text

    # 保存到文件
    out_path = Path("/data2/jrjiang/realname/code/event_detection/output/舆论分析报告.txt")
    out_path.write_text(report, encoding="utf-8")

    print("\n✅ 分析完成，报告已保存为：舆论分析报告.txt")
    print("📄 报告摘要预览：\n")
    print("\n".join(report.splitlines()[:25]))  # 只预览前 25 行


if __name__ == "__main__":
    main()
