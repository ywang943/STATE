"""
patient_context/augment_context.py
════════════════════════════════════════════════════════════════
KARE Step 2d: 用KG社区摘要增强患者上下文
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件：
  - data/base_contexts.jsonl（base_context.py 输出）
  - data/kg.json（build_kg.py 输出）
  - data/kg_communities.json（build_kg.py 输出）
  - data/sim_patients.json（sim_patient_ret.py 输出）

输出：data/augmented_contexts.jsonl
════════════════════════════════════════════════════════════════
"""
import sys
import json
import logging
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import (BASE_CONTEXT_FILE, KG_FILE, KG_COMMUNITIES_FILE,
                    SIM_PATIENTS_FILE, AUG_CONTEXT_FILE,
                    TOP_K_COMMUNITY, TOP_K_KG)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def community_relevance(patient_concepts: list, community: dict) -> float:
    comm_concepts = set(community["concepts"])
    patient_set   = set(patient_concepts)

    direct_overlap = len(patient_set & comm_concepts)

    combo_score = 0
    for cc in comm_concepts:
        if "+" in cc:
            parts = set(cc.split("+"))
            if parts & patient_set:
                combo_score += len(parts & patient_set) / len(parts)

    return direct_overlap + 0.5 * combo_score


def query_kg_herbs(kg: dict, bingxing_list: list, bingwei_list: list, top_k: int = 12) -> list:
    herb_scores = {}
    for bx in bingxing_list:
        for h, cnt in kg.get("bingxing", {}).get(bx, {}).items():
            herb_scores[h] = herb_scores.get(h, 0) + cnt
    for bw in bingwei_list:
        for h, cnt in kg.get("bingwei", {}).get(bw, {}).items():
            herb_scores[h] = herb_scores.get(h, 0) + cnt
    for bx in bingxing_list:
        for bw in bingwei_list:
            key = f"{bx}+{bw}"
            for h, cnt in kg.get("combined", {}).get(key, {}).items():
                herb_scores[h] = herb_scores.get(h, 0) + cnt * 1.5

    sorted_herbs = sorted(herb_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_herbs[:top_k]


def format_sim_patients(sim_list: list) -> str:
    lines = []
    for i, item in enumerate(sim_list, 1):
        ctx   = item.get("context", {})
        score = item.get("score", 0)
        lines.append(f"[相似病例{i}] (相似度={score:.3f})")
        if ctx.get("xianbing"):  lines.append(f"  现病史：{ctx['xianbing'][:80]}")
        if ctx.get("sizhen"):    lines.append(f"  四诊：{ctx['sizhen']}")
        if ctx.get("bingxing"): lines.append(f"  病性：{ctx['bingxing']}")
        if ctx.get("bingwei"):  lines.append(f"  病位：{ctx['bingwei']}")
        if ctx.get("bianzheng"):lines.append(f"  辨证：{ctx['bianzheng']}")
        if ctx.get("herbs_gt"): lines.append(f"  用药：{ctx['herbs_gt'][:60]}")
    return "\n".join(lines)


def main():
    contexts = []
    with open(BASE_CONTEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                contexts.append(json.loads(line))
    logger.info(f"Loaded {len(contexts)} base contexts")

    with open(KG_FILE, "r", encoding="utf-8") as f:
        kg = json.load(f)

    with open(KG_COMMUNITIES_FILE, "r", encoding="utf-8") as f:
        communities = json.load(f)
    logger.info(f"Loaded {len(communities)} KG communities")

    with open(SIM_PATIENTS_FILE, "r", encoding="utf-8") as f:
        sim_patients = json.load(f)
    logger.info(f"Loaded similar patients for {len(sim_patients)} queries")

    augmented = []

    for ctx in contexts:
        pid           = ctx["id"]
        bingxing_list = ctx["bingxing_list"]
        bingwei_list  = ctx["bingwei_list"]
        all_concepts  = bingxing_list + bingwei_list

        kg_herbs = query_kg_herbs(kg, bingxing_list, bingwei_list, top_k=TOP_K_KG)

        if communities:
            scored_communities = [
                (community_relevance(all_concepts, comm), comm)
                for comm in communities
            ]
            scored_communities.sort(key=lambda x: x[0], reverse=True)
            top_communities = [
                comm for score, comm in scored_communities[:TOP_K_COMMUNITY]
                if score > 0
            ]
        else:
            top_communities = []

        community_summaries = [c["summary"] for c in top_communities if "summary" in c]

        sim_list = sim_patients.get(pid, [])
        sim_text = format_sim_patients(sim_list) if sim_list else "无相似病例"

        kg_herbs_text = (
            "、".join([f"{h}(频次{int(s)})" for h, s in kg_herbs])
            if kg_herbs else "无"
        )
        community_text = (
            "\n".join([f"  [{i+1}] {s}" for i, s in enumerate(community_summaries)])
            if community_summaries else "  暂无相关社区摘要"
        )

        aug_ctx = dict(ctx)
        aug_ctx.update({
            "kg_herbs":            kg_herbs,
            "kg_herbs_text":       kg_herbs_text,
            "community_summaries": community_summaries,
            "community_text":      community_text,
            "sim_patients_text":   sim_text,
        })
        augmented.append(aug_ctx)

    with open(AUG_CONTEXT_FILE, "w", encoding="utf-8") as f:
        for aug in augmented:
            f.write(json.dumps(aug, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(augmented)} augmented contexts to {AUG_CONTEXT_FILE}")

    n_with_community = sum(1 for a in augmented if a.get("community_summaries"))
    logger.info(f"  Patients with KG community summaries: {n_with_community}/{len(augmented)}")


if __name__ == "__main__":
    main()
