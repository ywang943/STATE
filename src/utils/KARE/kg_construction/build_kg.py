"""
kg_construction/build_kg.py
════════════════════════════════════════════════════════════════
KARE Step 1: 知识图谱构建 + 层次社区检测 + 社区摘要生成
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

对应KARE原论文 Section 3.1:
  - 从患者EHR中提取concept-specific KG
  - 用社区检测把KG组织成层次社区
  - 用LLM为每个社区生成自然语言摘要（community summary）
  - 摘要将在Step 2中被检索注入患者上下文

TCM映射：
  - EHR concepts → 病性、病位（中医证候要素）
  - KG edges     → 证候要素 ─[共现]─ 中药
  - 社区          → 具有相似治法的证候-中药簇
  - 社区摘要      → 该证候群对应的中医治法与方药描述
════════════════════════════════════════════════════════════════
"""
import sys
import json
import logging
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import (PATIENTS_FILE, KG_FILE, KG_COMMUNITIES_FILE,
                    MIN_HERB_FREQ, COMMUNITY_MIN_SIZE,
                    LLM_TEMP_REASON, LLM_MAX_TOKENS)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def load_records(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def parse_list_field(value: str) -> list:
    """把 '心阴虚,肝郁,肾阳虚' 拆成列表，忽略空串"""
    return [x.strip() for x in str(value).split(",") if x.strip()]


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def build_raw_kg(records: list) -> dict:
    """
    返回结构：
    {
      "bingxing": { "心阴虚": {"麦冬": 12, ...}, ... },
      "bingwei":  { "心":     {"麦冬": 10, ...}, ... },
      "combined": { "心阴虚+心": {"麦冬": 9, ...}, ... }
    }
    """
    kg = {
        "bingxing": defaultdict(lambda: defaultdict(int)),
        "bingwei":  defaultdict(lambda: defaultdict(int)),
        "combined": defaultdict(lambda: defaultdict(int)),
    }

    for rec in records:
        m        = rec.get("metadata", {})
        herbs    = parse_list_field(m.get("中药名称", ""))
        bingxing = parse_list_field(m.get("病性(泛化)", ""))
        bingwei  = parse_list_field(m.get("病位(泛化)", ""))

        for bx in bingxing:
            for h in herbs:
                kg["bingxing"][bx][h] += 1

        for bw in bingwei:
            for h in herbs:
                kg["bingwei"][bw][h] += 1

        for bx in bingxing:
            for bw in bingwei:
                key = f"{bx}+{bw}"
                for h in herbs:
                    kg["combined"][key][h] += 1

    def filter_kg(sub: dict) -> dict:
        return {
            concept: {h: cnt for h, cnt in herbs.items() if cnt >= MIN_HERB_FREQ}
            for concept, herbs in sub.items()
        }

    kg_clean = {k: filter_kg(dict(v)) for k, v in kg.items()}
    logger.info(f"KG nodes: bingxing={len(kg_clean['bingxing'])}, "
                f"bingwei={len(kg_clean['bingwei'])}, "
                f"combined={len(kg_clean['combined'])}")
    return kg_clean


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def build_communities(kg: dict, threshold: float = 0.25) -> list:
    """
    把所有KG节点按herb集合相似度聚成社区。
    返回社区列表，每个社区包含：
      - concepts: 该社区涵盖的证候概念列表
      - herb_freq: {herb: total_freq}
      - top_herbs: 频次前15的中药
    """
    nodes = {}
    for layer in ("bingxing", "bingwei", "combined"):
        for concept, herbs in kg[layer].items():
            if len(herbs) >= COMMUNITY_MIN_SIZE:
                nodes[concept] = set(herbs.keys())

    node_list = list(nodes.items())
    if not node_list:
        return []

    communities = []

    for concept, herb_set in node_list:
        best_idx = -1
        best_sim = threshold

        for i, comm in enumerate(communities):
            sim = jaccard(herb_set, comm["herb_set"])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0:
            comm = communities[best_idx]
            comm["concepts"].append(concept)
            comm["herb_set"] |= herb_set
        else:
            communities.append({
                "id":       len(communities),
                "concepts": [concept],
                "herb_set": herb_set,
            })

    result = []
    for comm in communities:
        herb_freq = defaultdict(int)
        for concept in comm["concepts"]:
            for layer in ("bingxing", "bingwei", "combined"):
                if concept in kg[layer]:
                    for h, cnt in kg[layer][concept].items():
                        herb_freq[h] += cnt

        top_herbs = sorted(herb_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        result.append({
            "id":        comm["id"],
            "concepts":  comm["concepts"],
            "top_herbs": [h for h, _ in top_herbs],
            "herb_freq": dict(herb_freq),
        })

    logger.info(f"Built {len(result)} KG communities (threshold={threshold})")
    return result


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def generate_community_summary(community: dict, call_llm_fn) -> str:
    concepts_str  = "、".join(community["concepts"][:10])
    top_herbs_str = "、".join(community["top_herbs"][:12])

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名资深中医临床专家，精通辨证论治。"
                "请根据给定的中医证候群与常用中药，"
                "生成一段精炼的中医临床知识摘要，供临床决策参考。"
                "摘要应包括：该证候群的核心病机、治疗原则、代表性方药思路。"
                "字数控制在150字以内，语言简洁专业。"
            )
        },
        {
            "role": "user",
            "content": (
                f"【证候群概念】{concepts_str}\n"
                f"【常用中药（频次降序）】{top_herbs_str}\n\n"
                "请生成该证候群的中医临床知识摘要："
            )
        }
    ]

    try:
        summary = call_llm_fn(messages, temperature=LLM_TEMP_REASON, max_tokens=300)
        return summary
    except Exception as e:
        logger.warning(f"Community {community['id']} summary failed: {e}")
        return (f"该证候群涉及{concepts_str}，"
                f"临床常用中药包括{top_herbs_str}。"
                f"治疗时需根据具体证候配伍选药。")


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def main(use_llm_summary: bool = True):
    records = load_records(PATIENTS_FILE)

    logger.info("Building raw knowledge graph...")
    kg = build_raw_kg(records)
    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved KG to {KG_FILE}")

    logger.info("Running community detection...")
    communities = build_communities(kg)

    if use_llm_summary:
        from apis.openai_api import call_llm
        logger.info(f"Generating LLM summaries for {len(communities)} communities...")
        for i, comm in enumerate(communities):
            logger.info(f"  Community {i+1}/{len(communities)}: "
                        f"concepts={comm['concepts'][:3]}...")
            comm["summary"] = generate_community_summary(comm, call_llm)
    else:
        logger.info("Generating template summaries (no LLM)...")
        for comm in communities:
            concepts_str  = "、".join(comm["concepts"][:8])
            top_herbs_str = "、".join(comm["top_herbs"][:10])
            comm["summary"] = (
                f"该证候群涉及{concepts_str}，"
                f"临床常用中药包括{top_herbs_str}。"
            )

    with open(KG_COMMUNITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(communities, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(communities)} communities to {KG_COMMUNITIES_FILE}")

    logger.info("=" * 50)
    logger.info("KG construction complete.")
    logger.info(f"  Total communities: {len(communities)}")
    logger.info(f"  Avg herbs per community: "
                f"{sum(len(c['top_herbs']) for c in communities) / max(len(communities), 1):.1f}")


if __name__ == "__main__":
    USE_LLM = True
    main(use_llm_summary=USE_LLM)
