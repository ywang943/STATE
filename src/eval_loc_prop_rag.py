# ======================================
# ======================================
import json
import time
from typing import Dict, List

import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score


# ======================================
# OpenAI Client
# ======================================
client = OpenAI(api_key='YOUR_API_KEY')
MODEL_NAME = "gpt-4o"


# ======================================
# ======================================
def extract_json(text: str):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        return json.loads(text[l:r + 1])

    raise ValueError("No valid JSON found")


def llm_json_call(messages, retry=3, max_tokens=500):
    for _ in range(retry):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        raw = resp.choices[0].message.content
        try:
            return raw, extract_json(raw)
        except Exception:
            messages.append({
                "role": "user",
                "content": "上一次输出不是严格 JSON，请只输出 JSON，不要任何解释。"
            })
            time.sleep(1)
    raise RuntimeError("LLM JSON parse failed")


# ======================================
# ======================================
def char_tokenize(text: str):
    return list(text.replace(" ", ""))


def bleu_score(reference: str, hypothesis: str) -> float:
    return sentence_bleu(
        [char_tokenize(reference)],
        char_tokenize(hypothesis),
        smoothing_function=SmoothingFunction().method1
    )


# ======================================
# ======================================
def bertscore(pred: str, truth: str) -> float:
    if not pred or not truth:
        return 0.0
    P, R, F1 = bert_score([pred], [truth], lang="zh", verbose=False)
    return F1.mean().item()


def f1_score(pred: str, truth: str) -> float:
    p = {x.strip() for x in pred.replace("，", ",").split(",") if x.strip()}
    t = {x.strip() for x in truth.replace("，", ",").split(",") if x.strip()}

    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0

    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def jaccard_score(pred: str, truth: str) -> float:
    p = {x.strip() for x in pred.replace("，", ",").split(",") if x.strip()}
    t = {x.strip() for x in truth.replace("，", ",").split(",") if x.strip()}

    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0

    return len(p & t) / len(p | t)


# ======================================
# ======================================
RAG_KEYS_CORE = ["寒热", "虚实", "表里", "涉及可能脏腑", "涉及典型病机"]
RAG_KEYS_SURFACE = ["动态特征", "时间节律", "饮食相关", "情志相关", "消化表现"]


def build_rag_text(structured: Dict, keys: List[str]) -> str:
    return " ".join(
        f"{k}:{structured.get(k)}"
        for k in keys
        if structured.get(k) and structured.get(k) != "暂不确定"
    )


# ======================================
# ======================================
def build_faiss_index(train_path: str, keys: List[str]):
    embed_model = SentenceTransformer("moka-ai/m3e-base")
    docs, texts = [], []

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            s = row["structured"]

            bw, bx = s.get("病位", ""), s.get("病性", "")
            if bw == "无" and bx == "无":
                continue

            text = build_rag_text(s, keys)
            if not text.strip():
                continue

            texts.append(text)
            docs.append({"病位": bw, "病性": bx})

    emb = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return embed_model, docs, index


# ======================================
# ======================================
def rag_search(query, model, docs, index, topk):
    if not query.strip():
        return []

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, topk)

    return [
        f"【相似病例 {i+1} | score={D[0][i]:.4f}】\n病位：{docs[idx]['病位']}\n病性：{docs[idx]['病性']}"
        for i, idx in enumerate(I[0])
    ]


# ======================================
# Prompt
# ======================================
SYSTEM_PROMPT = "你是一名经验丰富的中医辨证论治医生。"

USER_PROMPT_TEMPLATE = """
下面是当前病例的中医【判断线索】（JSON）：
{features}

【相似病例参考】
{rag_context}

请推断当前病例的【病位】和【病性】。

输出要求：
只输出 JSON，格式如下：
{{
  "病位": "",
  "病性": ""
}}
"""


# ======================================
# ======================================
def run_experiment(test_path, train_path, max_cases=20):
    embed_model, core_docs, core_index = build_faiss_index(train_path, RAG_KEYS_CORE)
    _, surf_docs, surf_index = build_faiss_index(train_path, RAG_KEYS_SURFACE)

    metrics = {
        "bw": {"bleu": [], "bert": [], "f1": [], "jac": []},
        "bx": {"bleu": [], "bert": [], "f1": [], "jac": []}
    }

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_cases:
                break

            s = json.loads(line)["structured"]
            true_bw, true_bx = s.get("病位", ""), s.get("病性", "")
            if true_bw == "无" or true_bx == "无":
                continue

            rag_context = ""
            rag_context += "\n".join(
                rag_search(build_rag_text(s, RAG_KEYS_CORE), embed_model, core_docs, core_index, 5)
            )
            rag_context += "\n"
            rag_context += "\n".join(
                rag_search(build_rag_text(s, RAG_KEYS_SURFACE), embed_model, surf_docs, surf_index, 3)
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    features=json.dumps(s, ensure_ascii=False, indent=2),
                    rag_context=rag_context
                )}
            ]

            raw, pred = llm_json_call(messages)
            pred_bw, pred_bx = pred.get("病位", ""), pred.get("病性", "")

            metrics["bw"]["bleu"].append(bleu_score(true_bw, pred_bw))
            metrics["bw"]["bert"].append(bertscore(pred_bw, true_bw))
            metrics["bw"]["f1"].append(f1_score(pred_bw, true_bw))
            metrics["bw"]["jac"].append(jaccard_score(pred_bw, true_bw))

            metrics["bx"]["bleu"].append(bleu_score(true_bx, pred_bx))
            metrics["bx"]["bert"].append(bertscore(pred_bx, true_bx))
            metrics["bx"]["f1"].append(f1_score(pred_bx, true_bx))
            metrics["bx"]["jac"].append(jaccard_score(pred_bx, true_bx))

            print("=" * 80)
            print("GT:", true_bw, true_bx)
            print("PR:", pred_bw, pred_bx)

    print("\n====== Overall Evaluation ======")
    for k in ["bw", "bx"]:
        print(f"\n[{ '病位' if k=='bw' else '病性' }]")
        for m in metrics[k]:
            print(f"{m.upper():8}: {np.mean(metrics[k][m]):.4f}")


# ======================================
# ======================================
if __name__ == "__main__":
    run_experiment(
        test_path="test_jia_structured_filtered_5.jsonl",
        train_path="train_jia_structured_filtered_5.jsonl",
        max_cases=20
    )
