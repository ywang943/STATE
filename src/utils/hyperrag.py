# ======================================
# ======================================
import json
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer


# ======================================
# ======================================
AXIS_FIELDS = ["寒热", "虚实", "表里"]

SEMANTIC_FIELDS = [
    "涉及可能脏腑",
    "涉及典型病机",
    "动态特征",
    "时间节律",
    "饮食相关",
    "情志相关",
    "消化表现"
]

AXIS_NEIGHBORS = {
    "寒热": {
        "寒": ["寒", "偏寒"],
        "偏寒": ["寒", "偏寒"],
        "中性": ["中性"],
        "偏热": ["偏热", "热"],
        "热": ["偏热", "热"]
    },
    "虚实": {
        "虚": ["虚", "偏虚"],
        "偏虚": ["虚", "偏虚"],
        "偏实": ["偏实", "实"],
        "实": ["偏实", "实"]
    },
    "表里": {
        "表": ["表", "偏表"],
        "偏表": ["表", "偏表"],
        "偏里": ["偏里", "里"],
        "里": ["偏里", "里"]
    }
}


# ======================================
# ======================================
def valid_value(v: str) -> bool:
    return v and v not in {"无", "暂不确定"}


def make_entity(field: str, value: str) -> str:
    return f"{field}={value}"


def load_hyperedges(path: str) -> List[dict]:
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            edges.append(json.loads(line))
    return edges


# ======================================
# ======================================
def build_indexes(edges: List[dict]):
    embed_model = SentenceTransformer("moka-ai/m3e-base")

    field_to_entities: Dict[str, List[str]] = {}
    field_to_index: Dict[str, faiss.Index] = {}

    for field in SEMANTIC_FIELDS:
        entity_set = set()
        for e in edges:
            for ent in e["entities"]:
                if ent.startswith(field + "="):
                    _, v = ent.split("=", 1)
                    if valid_value(v):
                        entity_set.add(ent)

        if not entity_set:
            continue

        entity_list = sorted(entity_set)
        embs = embed_model.encode(
            entity_list,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        field_to_entities[field] = entity_list
        field_to_index[field] = index

    # ---- Summary Index ----
    summaries = [e["总结描述"] for e in edges]
    summary_embs = embed_model.encode(
        summaries,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    summary_index = faiss.IndexFlatIP(summary_embs.shape[1])
    summary_index.add(summary_embs)

    return embed_model, field_to_entities, field_to_index, summary_index


# ======================================
# Step A: Query → Entity
# ======================================
def _retrieve_entities(
    structured: dict,
    embed_model,
    field_to_entities,
    field_to_index,
    topk_per_field: int
) -> List[str]:
    selected = []

    for field in AXIS_FIELDS:
        v = structured.get(field, "")
        if not valid_value(v):
            continue
        for nv in AXIS_NEIGHBORS[field].get(v, []):
            selected.append(make_entity(field, nv))

    for field in SEMANTIC_FIELDS:
        v = structured.get(field, "")
        if not valid_value(v) or field not in field_to_entities:
            continue

        q_text = make_entity(field, v)
        q_emb = embed_model.encode(
            [q_text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        _, I = field_to_index[field].search(q_emb, topk_per_field)
        for idx in I[0]:
            selected.append(field_to_entities[field][idx])

    return list(set(selected))


# ======================================
# Step B: Entity → Hyperedge（hard overlap）
# ======================================
def _search_by_entity(
    selected_entities: List[str],
    edges: List[dict],
    topk: int
) -> List[dict]:
    results = []
    qset = set(selected_entities)
    q_size = len(qset)

    if q_size == 0:
        return []

    for e in edges:
        eset = set(e["entities"])
        hit = qset & eset
        if not hit:
            continue

        score = len(hit) / q_size
        conf = float(e.get("置信分数", 1.0))

        results.append({
            "edge_id": e["edge_id"],
            "edge": e,
            "entity_score": score * 2,
            "summary_score": None,
            "source": "entity",
            "hit_entities": sorted(hit)
        })

    results.sort(key=lambda x: x["entity_score"], reverse=True)
    return results[:topk]


# ======================================
# Step C: Summary → Hyperedge
# ======================================
def _search_by_summary(
    structured: dict,
    edges: List[dict],
    embed_model,
    summary_index,
    topk: int
) -> List[dict]:
    q = structured.get("总结描述", "")
    if not q:
        return []

    q_emb = embed_model.encode(
        [q],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    D, I = summary_index.search(q_emb, topk)
    results = []

    for idx, score in zip(I[0], D[0]):
        e = edges[idx]
        conf = float(e.get("置信分数", 1.0))
        results.append({
            "edge_id": e["edge_id"],
            "edge": e,
            "entity_score": None,
            "summary_score": float(score) * conf,
            "source": "summary",
            "hit_entities": []
        })

    return results


# ======================================
# Merge / Dedup
# ======================================
def _dedup_and_merge(
    entity_hits: List[dict],
    summary_hits: List[dict]
) -> List[dict]:
    merged: Dict[str, dict] = {}

    def add(c):
        eid = c["edge_id"]
        if eid not in merged:
            merged[eid] = {
                "edge": c["edge"],
                "entity_score": None,
                "summary_score": None,
                "sources": set(),
                "hit_entities": set()
            }
        merged[eid]["sources"].add(c["source"])
        merged[eid]["hit_entities"].update(c["hit_entities"])
        if c["source"] == "entity":
            merged[eid]["entity_score"] = c["entity_score"]
        else:
            merged[eid]["summary_score"] = c["summary_score"]

    for c in entity_hits:
        add(c)
    for c in summary_hits:
        add(c)

    results = []
    for eid, m in merged.items():
        scores = [s for s in [m["entity_score"], m["summary_score"]] if s is not None]
        results.append({
            "edge_id": eid,
            "edge": m["edge"],
            "entity_score": m["entity_score"],
            "summary_score": m["summary_score"],
            "sources": list(m["sources"]),
            "hit_entities": sorted(m["hit_entities"]),
            "final_score": max(scores)
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# ======================================
# ======================================
def rag_search(
    structured_query: dict,
    edges: List[dict],
    embed_model,
    field_to_entities,
    field_to_index,
    summary_index,
    topk_entity: int = 5,
    topk_summary: int = 5,
    topk_per_field: int = 2
) -> List[dict]:

    selected_entities = _retrieve_entities(
        structured_query,
        embed_model,
        field_to_entities,
        field_to_index,
        topk_per_field
    )

    cand_entity = _search_by_entity(
        selected_entities, edges, topk_entity
    )

    cand_summary = _search_by_summary(
        structured_query, edges, embed_model, summary_index, topk_summary
    )

    return _dedup_and_merge(cand_entity, cand_summary)


# ======================================
# ======================================
if __name__ == "__main__":
    edges = load_hyperedges("hyperedges.jsonl")

    (
        embed_model,
        field_to_entities,
        field_to_index,
        summary_index
    ) = build_indexes(edges)

    with open("../test_wei_structured.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(next(f))["structured"]

    results = rag_search(
        structured_query=sample,
        edges=edges,
        embed_model=embed_model,
        field_to_entities=field_to_entities,
        field_to_index=field_to_index,
        summary_index=summary_index,
        topk_entity=5,
        topk_summary=5,
        topk_per_field=2
    )

    print(results)

