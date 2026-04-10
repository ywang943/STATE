"""
build_embeddings.py
===================
Step 2: 用 moka-ai/m3e-base 对所有病历的「现病史」预计算 embedding
并保存到 data/embeddings.npz，供检索时使用。

首次运行会自动从 HuggingFace 下载 m3e-base 模型（约 800MB）。
后续运行直接加载本地缓存，速度很快。
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "moka-ai/m3e-base"
DATA_PATH  = "data/train_wei.jsonl"
EMB_PATH   = "data/embeddings.npz"


def load_data(data_path: str) -> list[dict]:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_query_text(record: dict) -> str:
    """
    决定用哪个字段做embedding的文本。
    这里用「现病史」作为主要输入，如果还有「临床表现(标杆词)」可以拼接进去增强语义。
    """
    meta = record.get("metadata", {})
    parts = []

    xbs = meta.get("现病史", "").strip()
    if xbs:
        parts.append(xbs)

    clinical = meta.get("临床表现(标杆词)", "").strip()
    if clinical:
        parts.append(f"临床表现：{clinical}")

    sizhen = meta.get("四诊(规范)", "").strip()
    if sizhen:
        parts.append(f"四诊：{sizhen}")

    return "。".join(parts) if parts else record.get("text", "")


def build_embeddings(records: list[dict], model: SentenceTransformer) -> tuple[np.ndarray, list[str]]:
    """
    返回：
      embeddings: shape (N, 768)
      ids: 每条记录的id列表，与embeddings行对齐
    """
    texts = [get_query_text(r) for r in records]
    ids   = [r["id"] for r in records]

    print(f"[build_embeddings] 开始编码 {len(texts)} 条现病史...")
    print(f"  示例文本: {texts[0][:80]}...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"[build_embeddings] 编码完成，shape: {embeddings.shape}")
    return embeddings, ids


def save_embeddings(embeddings: np.ndarray, ids: list[str], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, embeddings=embeddings, ids=np.array(ids))
    print(f"[build_embeddings] Embeddings已保存到: {output_path}")


def load_embeddings(emb_path: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(emb_path, allow_pickle=True)
    return data["embeddings"], data["ids"].tolist()


if __name__ == "__main__":
    print(f"[build_embeddings] 加载模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    records = load_data(DATA_PATH)
    embeddings, ids = build_embeddings(records, model)
    save_embeddings(embeddings, ids, EMB_PATH)

    print("\n[验证] 前5条ID：", ids[:5])
    print("[验证] Embedding维度：", embeddings.shape[1])
