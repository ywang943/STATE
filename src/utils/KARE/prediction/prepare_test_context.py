"""
prepare_test_context.py
════════════════════════════════════════════════════════════════
Step 2e（仅 Test）: 对测试集构建增强上下文
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件（Step 1-2 跑完后已有）：
  - data/train.jsonl          ← 已有
  - data/kg.json              ← build_kg.py 输出
  - data/kg_communities.json  ← build_kg.py 输出
  - data/embeddings.npz       ← get_emb.py 输出（train 的 embedding）
  - data/base_contexts.jsonl  ← base_context.py 输出（train 的上下文）
  - data/test.jsonl           ← ★ 你的测试集放这里

输出：
  - data/test_base_contexts.jsonl   ← test 的基础上下文
  - data/test_augmented_contexts.jsonl  ← test 的增强上下文（含KG+相似患者）

关键设计：
  - test 患者的相似患者从【train embedding库】里检索，不包含 test 自身
    （这样才不会数据泄露）
  - KG 和社区摘要也全部来自 train，test 只是查询方
════════════════════════════════════════════════════════════════
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import (
    TEST_FILE, TEST_BASE_CONTEXT_FILE, TEST_AUG_CONTEXT_FILE,
    KG_FILE, KG_COMMUNITIES_FILE,
    EMBEDDINGS_FILE, BASE_CONTEXT_FILE,
    EMB_MODEL_NAME, TOP_K_SIM, TOP_K_KG, TOP_K_COMMUNITY,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def build_base_context(record: dict) -> dict:
    pid = record.get("id", "")
    m   = record.get("metadata", {})

    xianbing  = m.get("现病史",     "").strip()
    sizhen    = m.get("四诊(规范)", "").strip()
    bingxing  = m.get("病性(泛化)", "").strip()
    bingwei   = m.get("病位(泛化)", "").strip()
    bianzheng = m.get("中医辨证",   "").strip()
    zy_diag   = m.get("中医诊断",   "").strip()
    xy_diag   = m.get("西医诊断",   "").strip()
    herbs_gt  = m.get("中药名称",   "").strip()

    parts = []
    if xianbing:  parts.append(f"现病史：{xianbing}")
    if sizhen:    parts.append(f"四诊（舌脉等）：{sizhen}")
    if bingxing:  parts.append(f"病性：{bingxing}")
    if bingwei:   parts.append(f"病位：{bingwei}")
    if bianzheng: parts.append(f"中医辨证：{bianzheng}")
    if zy_diag:   parts.append(f"中医诊断：{zy_diag}")
    if xy_diag:   parts.append(f"西医诊断：{xy_diag}")

    return {
        "id":            pid,
        "context_text":  "；".join(parts),
        "xianbing":      xianbing,
        "sizhen":        sizhen,
        "bingxing":      bingxing,
        "bingwei":       bingwei,
        "bianzheng":     bianzheng,
        "zy_diag":       zy_diag,
        "xy_diag":       xy_diag,
        "herbs_gt":      herbs_gt,
        "bingxing_list": [x.strip() for x in bingxing.split(",") if x.strip()],
        "bingwei_list":  [x.strip() for x in bingwei.split(",")  if x.strip()],
        "herbs_gt_list": [x.strip() for x in herbs_gt.split(",") if x.strip()],
    }


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
    return sorted(herb_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


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


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def main():
    test_records = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))
    logger.info(f"Loaded {len(test_records)} test records from {TEST_FILE}")

    test_contexts = [build_base_context(r) for r in test_records]
    test_contexts = [c for c in test_contexts if c["xianbing"]]
    logger.info(f"Valid test contexts: {len(test_contexts)}")

    with open(TEST_BASE_CONTEXT_FILE, "w", encoding="utf-8") as f:
        for ctx in test_contexts:
            f.write(json.dumps(ctx, ensure_ascii=False) + "\n")
    logger.info(f"Saved test base contexts to {TEST_BASE_CONTEXT_FILE}")

    logger.info(f"Loading train embeddings from {EMBEDDINGS_FILE}...")
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    train_embeddings = data["embeddings"].astype(np.float32)
    train_ids        = data["ids"].tolist()
    logger.info(f"Train embeddings: {train_embeddings.shape}, {len(train_ids)} patients")

    train_ctx_map = {}
    with open(BASE_CONTEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                train_ctx_map[c["id"]] = c

    logger.info(f"Loading embedding model: {EMB_MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL_NAME)

    test_texts = [c["context_text"] for c in test_contexts]
    logger.info(f"Encoding {len(test_texts)} test contexts...")
    test_embeddings = model.encode(
        test_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    test_embeddings = np.array(test_embeddings, dtype=np.float32)

    try:
        import faiss
        logger.info("Using FAISS for retrieval")
        index = faiss.IndexFlatIP(train_embeddings.shape[1])
        index.add(train_embeddings)
        use_faiss = True
    except ImportError:
        logger.info("FAISS not available, using sklearn cosine similarity")
        use_faiss = False

    with open(KG_FILE, "r", encoding="utf-8") as f:
        kg = json.load(f)
    with open(KG_COMMUNITIES_FILE, "r", encoding="utf-8") as f:
        communities = json.load(f)
    logger.info(f"Loaded KG and {len(communities)} communities")

    augmented = []

    for i, ctx in enumerate(test_contexts):
        pid           = ctx["id"]
        bingxing_list = ctx["bingxing_list"]
        bingwei_list  = ctx["bingwei_list"]
        all_concepts  = bingxing_list + bingwei_list

        query_emb = test_embeddings[i:i + 1]

        if use_faiss:
            scores, indices = index.search(query_emb, TOP_K_SIM)
            sim_list = [
                {"id": train_ids[idx], "score": float(scores[0][j]),
                 "context": train_ctx_map.get(train_ids[idx], {})}
                for j, idx in enumerate(indices[0])
            ]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(query_emb, train_embeddings)[0]
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            sim_list = [
                {"id": train_ids[idx], "score": float(score),
                 "context": train_ctx_map.get(train_ids[idx], {})}
                for idx, score in ranked[:TOP_K_SIM]
            ]

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

        kg_herbs_text = (
            "、".join([f"{h}(频次{int(s)})" for h, s in kg_herbs])
            if kg_herbs else "无"
        )
        community_text = (
            "\n".join([f"  [{j+1}] {s}" for j, s in enumerate(community_summaries)])
            if community_summaries else "  暂无相关社区摘要"
        )
        sim_text = format_sim_patients(sim_list) if sim_list else "无相似病例"

        aug_ctx = dict(ctx)
        aug_ctx.update({
            "kg_herbs":            kg_herbs,
            "kg_herbs_text":       kg_herbs_text,
            "community_summaries": community_summaries,
            "community_text":      community_text,
            "sim_patients_text":   sim_text,
        })
        augmented.append(aug_ctx)

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(test_contexts)}")

    with open(TEST_AUG_CONTEXT_FILE, "w", encoding="utf-8") as f:
        for aug in augmented:
            f.write(json.dumps(aug, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(augmented)} test augmented contexts to {TEST_AUG_CONTEXT_FILE}")

    n_with_sim = sum(1 for a in augmented if a.get("sim_patients_text") != "无相似病例")
    n_with_comm = sum(1 for a in augmented if a.get("community_summaries"))
    logger.info(f"  With similar patients: {n_with_sim}/{len(augmented)}")
    logger.info(f"  With community summaries: {n_with_comm}/{len(augmented)}")


if __name__ == "__main__":
    main()
