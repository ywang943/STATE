"""
retriever.py
============
Step 3: 给定新患者的现病史，从历史病例库检索 top-K 最相似病例。

核心逻辑：
  1. 把新患者的现病史用 m3e-base 编码成向量
  2. 与预存的全库 embeddings 做余弦相似度（已归一化 → 直接点积）
  3. 返回 top-K 相似病例的完整信息

这对应原始 MedRAG 的 EHR Retrieval 模块。
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from build_embeddings import get_query_text, MODEL_NAME


class TCMRetriever:
    def __init__(
        self,
        data_path: str    = "data/train_jia.jsonl",
        emb_path:  str    = "data/embeddings.npz",
        model_name: str   = MODEL_NAME,
    ):
        print("[Retriever] 加载模型和数据...")

        self.model = SentenceTransformer(model_name)

        data = np.load(emb_path, allow_pickle=True)
        self.embeddings = data["embeddings"]   # shape (N, 768)
        self.ids = data["ids"].tolist()

        self.records = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.records[rec["id"]] = rec

        print(f"[Retriever] 就绪，数据库共 {len(self.ids)} 条病历")

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        exclude_id: str = None,
    ) -> list[dict]:
        """
        检索最相似的 top_k 条历史病例。

        返回列表，每个元素是 dict，包含：
          - id, similarity, 现病史, 中药名称, 病性, 病位 等
        """
        query_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # shape (1, 768)

        sims = (self.embeddings @ query_emb.T).flatten()  # shape (N,)

        sorted_indices = np.argsort(sims)[::-1]

        results = []
        for idx in sorted_indices:
            pid = self.ids[idx]
            if pid == exclude_id:
                continue
            if pid not in self.records:
                continue

            rec = self.records[pid]
            meta = rec.get("metadata", {})
            results.append({
                "id": pid,
                "similarity": float(sims[idx]),
                "现病史": meta.get("现病史", ""),
                "中药名称": meta.get("中药名称", ""),
                "病性": meta.get("病性(泛化)", ""),
                "病位": meta.get("病位(泛化)", ""),
                "中医辨证": meta.get("中医辨证", ""),
                "中医诊断": meta.get("中医诊断", ""),
            })

            if len(results) >= top_k:
                break

        return results


def format_retrieved_cases(cases: list[dict]) -> str:
    """把检索到的病例格式化成适合塞进 Prompt 的文字"""
    lines = []
    for i, c in enumerate(cases, 1):
        lines.append(f"【参考病例 {i}】（相似度: {c['similarity']:.3f}）")
        lines.append(f"  现病史: {c['现病史']}")
        if c['病性']:
            lines.append(f"  病性: {c['病性']}")
        if c['病位']:
            lines.append(f"  病位: {c['病位']}")
        if c['中医辨证']:
            lines.append(f"  中医辨证: {c['中医辨证']}")
        lines.append(f"  处方中药: {c['中药名称']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    retriever = TCMRetriever()

    query = "服用中药后有睡意，但入睡仍困难，服用安眠药入睡，眠浅，梦多，夜尿一次，较前明显减少。"
    print(f"\n[测试] 查询: {query[:60]}...\n")

    results = retriever.retrieve(query, top_k=3)
    print(format_retrieved_cases(results))
