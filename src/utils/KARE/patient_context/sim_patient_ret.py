"""
patient_context/sim_patient_ret.py
════════════════════════════════════════════════════════════════
KARE Step 2c: 相似患者检索
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件：
  - data/base_contexts.jsonl（base_context.py 输出）
  - data/embeddings.npz（get_emb.py 输出）

输出：data/sim_patients.json

使用 FAISS（优先）或 sklearn cosine 相似度作为 fallback。
Leave-one-out：检索时自动排除自身。
════════════════════════════════════════════════════════════════
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import EMBEDDINGS_FILE, BASE_CONTEXT_FILE, SIM_PATIENTS_FILE, TOP_K_SIM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    ids = data["ids"].tolist()
    logger.info(f"Loaded embeddings: {embeddings.shape}, {len(ids)} patients")

    ctx_map = {}
    with open(BASE_CONTEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                ctx_map[c["id"]] = c

    try:
        import faiss
        logger.info("Using FAISS for retrieval")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        use_faiss = True
    except ImportError:
        logger.info("FAISS not available, using sklearn cosine similarity")
        use_faiss = False

    sim_results = {}

    for i, pid in enumerate(ids):
        query_emb = embeddings[i:i + 1]

        if use_faiss:
            scores, indices = index.search(query_emb, TOP_K_SIM + 1)
            candidates = [(ids[idx], float(scores[0][j]))
                          for j, idx in enumerate(indices[0])
                          if ids[idx] != pid][:TOP_K_SIM]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(query_emb, embeddings)[0]
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            candidates = [(ids[idx], float(score))
                          for idx, score in ranked
                          if ids[idx] != pid][:TOP_K_SIM]

        sim_results[pid] = [
            {"id": cid, "score": score, "context": ctx_map.get(cid, {})}
            for cid, score in candidates
        ]

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(ids)}")

    with open(SIM_PATIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(sim_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved similar patients index to {SIM_PATIENTS_FILE}")
    logger.info(f"  Total queries: {len(sim_results)}")


if __name__ == "__main__":
    main()
