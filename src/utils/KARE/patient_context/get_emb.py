"""
patient_context/get_emb.py
════════════════════════════════════════════════════════════════
KARE Step 2b: 计算患者上下文 Embedding
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件：data/base_contexts.jsonl 已存在（先跑 base_context.py）

输出：data/embeddings.npz
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

from config import BASE_CONTEXT_FILE, EMBEDDINGS_FILE, EMB_MODEL_NAME

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    contexts = []
    with open(BASE_CONTEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                contexts.append(json.loads(line))
    logger.info(f"Loaded {len(contexts)} base contexts")

    ids   = [c["id"] for c in contexts]
    texts = [c["context_text"] for c in contexts]

    logger.info(f"Loading embedding model: {EMB_MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL_NAME)

    logger.info("Encoding patient contexts...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    np.savez(EMBEDDINGS_FILE, embeddings=embeddings, ids=np.array(ids))
    logger.info(f"Saved embeddings to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
