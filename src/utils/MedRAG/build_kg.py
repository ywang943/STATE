"""
build_kg.py
===========
Step 1: 从病历数据集构建「中药知识图谱」

知识图谱结构（对应原始MedRAG的Disease KG）：
  - 节点：病性、病位、中药
  - 边：(病性, 病位) → 常用中药列表 + 频次统计

输出：kg.json，供 main.py 查询使用
"""

import json
import os
from collections import defaultdict


def load_data(data_path: str) -> list[dict]:
    """加载 JSONL 格式数据（每行一个JSON对象）"""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[build_kg] 加载了 {len(records)} 条病历")
    return records


def build_kg(records: list[dict]) -> dict:
    """
    构建知识图谱，三层结构：
      1. 病性 → 关联中药（直接统计）
      2. 病位 → 关联中药
      3. (病性, 病位) 组合 → 关联中药（最精准）
      4. 单味中药 → 出现频次（全局统计，用于兜底）
    """
    bingxing_herb = defaultdict(lambda: defaultdict(int))
    bingwei_herb  = defaultdict(lambda: defaultdict(int))
    combo_herb    = defaultdict(lambda: defaultdict(int))
    global_herb   = defaultdict(int)

    for rec in records:
        meta = rec.get("metadata", {})

        bingxing_str = meta.get("病性(泛化)", "")
        bingxing_list = [x.strip() for x in bingxing_str.split(",") if x.strip()]

        bingwei_str = meta.get("病位(泛化)", "")
        bingwei_list = [x.strip() for x in bingwei_str.split(",") if x.strip()]

        herb_str = meta.get("中药名称", "")
        herb_list = [x.strip() for x in herb_str.split(",") if x.strip()]

        if not herb_list:
            continue

        for herb in herb_list:
            global_herb[herb] += 1

        for bx in bingxing_list:
            for herb in herb_list:
                bingxing_herb[bx][herb] += 1

        for bw in bingwei_list:
            for herb in herb_list:
                bingwei_herb[bw][herb] += 1

        for bx in bingxing_list:
            for bw in bingwei_list:
                key = f"{bx}|{bw}"
                for herb in herb_list:
                    combo_herb[key][herb] += 1

    kg = {
        "bingxing_herb": {k: dict(v) for k, v in bingxing_herb.items()},
        "bingwei_herb":  {k: dict(v) for k, v in bingwei_herb.items()},
        "combo_herb":    {k: dict(v) for k, v in combo_herb.items()},
        "global_herb":   dict(global_herb),
        "meta": {
            "total_records": len(records),
            "unique_herbs": len(global_herb),
            "unique_bingxing": len(bingxing_herb),
            "unique_bingwei": len(bingwei_herb),
        }
    }

    print(f"[build_kg] KG统计：")
    print(f"  总病历数: {kg['meta']['total_records']}")
    print(f"  独立中药数: {kg['meta']['unique_herbs']}")
    print(f"  独立病性数: {kg['meta']['unique_bingxing']}")
    print(f"  独立病位数: {kg['meta']['unique_bingwei']}")
    return kg


def query_kg(kg: dict, bingxing_list: list[str], bingwei_list: list[str], top_k: int = 15) -> list[tuple]:
    """
    根据病性+病位查询KG，返回推荐中药列表（按综合得分排序）

    得分策略：
      combo命中 × 3 + 病性命中 × 2 + 病位命中 × 1
    """
    herb_score = defaultdict(float)

    for bx in bingxing_list:
        for bw in bingwei_list:
            key = f"{bx}|{bw}"
            if key in kg["combo_herb"]:
                for herb, cnt in kg["combo_herb"][key].items():
                    herb_score[herb] += cnt * 3

    for bx in bingxing_list:
        if bx in kg["bingxing_herb"]:
            for herb, cnt in kg["bingxing_herb"][bx].items():
                herb_score[herb] += cnt * 2

    for bw in bingwei_list:
        if bw in kg["bingwei_herb"]:
            for herb, cnt in kg["bingwei_herb"][bw].items():
                herb_score[herb] += cnt * 1

    sorted_herbs = sorted(herb_score.items(), key=lambda x: x[1], reverse=True)
    return sorted_herbs[:top_k]


def save_kg(kg: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    print(f"[build_kg] KG已保存到: {output_path}")


if __name__ == "__main__":
    DATA_PATH = "data/train_wei.jsonl"
    KG_PATH   = "data/kg.json"

    records = load_data(DATA_PATH)
    kg = build_kg(records)
    save_kg(kg, KG_PATH)

    print("\n[测试] 查询病性=心阴虚,肝郁 / 病位=心,肝 的KG推荐中药：")
    results = query_kg(kg, ["心阴虚", "肝郁"], ["心", "肝"], top_k=10)
    for herb, score in results:
        print(f"  {herb}: {score:.1f}")
