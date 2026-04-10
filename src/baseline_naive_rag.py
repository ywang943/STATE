# ======================================
# ======================================
import json
import re
from typing import List

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from pathlib import Path
# ======================================
# ======================================
GPT_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
GPT_API_KEY = "874e40a79e924dd8a3695e20b619aaf7121b91563f3b4948af29b5fd10cdffc0"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": GPT_API_KEY
}

MODEL_NAME = "gpt-4"
print("使用的模型：", MODEL_NAME)

TOPK_HERB = 10
TOPK_RAG = 5
MAX_TEST_CASE = 100
DEBUG = True

SCRIPT_DIR = Path(__file__).resolve().parent
HERB_CATEGORY_DIR = SCRIPT_DIR / "herb_category"
CATEGORY_CACHE_PATH = SCRIPT_DIR / "herb_category_llm.json"

ENABLE_LLM_CATEGORY = True
DEFAULT_CATEGORY = "无分类"

HERB_STOPWORDS = {"无", "暂无", "无推荐", "无推荐中药"}

# ======================================
# ======================================
with open("unique_herbs.txt", "r", encoding="utf-8") as f:
    herb_vocab = [line.strip() for line in f if line.strip()]
HERB_VOCAB_TEXT = "，".join(herb_vocab)


# ======================================
# ======================================
def extract_herbs_from_output(text, topk=10):
    herbs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("中药："):
            content = line[len("中药："):]
            parts = re.split(r"[，,、\s]+", content)
            herbs = [p.strip() for p in parts if p.strip()]
            break
    herbs = herbs[:topk]
    herbs += [""] * (topk - len(herbs))
    return herbs


def parse_truth_herbs(text):
    herbs = re.split(r"[，,、\s]+", text)
    return [h.strip() for h in herbs if h.strip()]


def p_at_k(pred, truth_set, k=10):
    return sum(1 for h in pred[:k] if h in truth_set) / k



def normalize_herb_name(name: str) -> str:
    return name.strip()

# ======================================
# ======================================
def load_category_index(root_dir: Path):
    herb_to_categories = {}
    categories = set()
    if not root_dir.exists():
        return {}, []

    for path in root_dir.rglob("*.txt"):
        category = path.parent.name
        herb = normalize_herb_name(path.stem)
        categories.add(category)
        herb_to_categories.setdefault(herb, set()).add(category)

    herb_to_category = {
        herb: sorted(list(cats))[0]
        for herb, cats in herb_to_categories.items()
    }
    return herb_to_category, sorted(categories)


def load_category_cache(path: Path, categories: List[str]):
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    cache = {}
    for herb, cats in raw.items():
        herb = normalize_herb_name(herb)
        if isinstance(cats, list):
            for c in cats:
                if c in categories:
                    cache[herb] = c
                    break
        elif cats in categories:
            cache[herb] = cats
    return cache


def save_category_cache(path: Path, cache: dict):
    payload = {k: [v] for k, v in cache.items()}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def f1_cl_score(pred_categories, truth_categories):
    """
    基于「大类集合」的 Precision / Recall / F1
    """
    P = set(pred_categories)
    G = set(truth_categories)

    if not P or not G:
        return 0.0, 0.0, 0.0

    inter = len(P & G)
    precision = inter / len(P)
    recall = inter / len(G)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def classify_herbs_via_llm(herbs, categories, fallback):
    system = (
        "你是中药分类助手。"
        "请从给定的大类中为每味中药选择最合适的1个类别。"
        "只能输出 JSON。"
    )
    category_text = "、".join(categories)
    results = {}

    for herb in herbs:
        user = (
            f"大类列表：{category_text}\n"
            f"中药：{herb}\n"
            "输出示例：{\"中药名\": \"大类\"}"
        )
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0,
            "max_tokens": 200
        }
        try:
            r = requests.post(
                GPT_API_URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=300
            )
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            obj = json.loads(re.search(r"\{.*\}", text, re.S).group())
            cat = list(obj.values())[0]
            results[herb] = cat if cat in categories else fallback
        except Exception:
            results[herb] = fallback

    return results


def ensure_herbs_categorized(
    herbs, herb_to_category, categories, cache_path, fallback
):
    missing = [
        h for h in herbs
        if h and h not in herb_to_category
    ]
    if not missing:
        return herb_to_category

    if ENABLE_LLM_CATEGORY:
        new_map = classify_herbs_via_llm(missing, categories, fallback)
    else:
        new_map = {h: fallback for h in missing}

    herb_to_category.update(new_map)

    cache = load_category_cache(cache_path, categories)
    cache.update(new_map)
    save_category_cache(cache_path, cache)

    return herb_to_category

def herbs_to_categories(
    herbs, herb_to_category, categories, cache_path, fallback
):
    herbs = [normalize_herb_name(h) for h in herbs if h]
    herb_to_category = ensure_herbs_categorized(
        herbs, herb_to_category, categories, cache_path, fallback
    )
    return [herb_to_category.get(h, fallback) for h in herbs]

# ======================================
# Naive RAG
# ======================================
def build_naive_query(structured: dict) -> str:
    parts = []
    for k, v in structured.items():
        if k == "总结描述":
            continue
        if isinstance(v, list):
            v = "、".join(v)
        parts.append(f"{k}：{v}")
    return "\n".join(parts)


def load_naive_rag_corpus(train_path):
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            structured = row["structured"]
            text = build_naive_query(structured)
            texts.append(text)

    #print("=" * 60)
    #print(texts[0][:500])
    return texts


def naive_rag_search(query_text, corpus_embeds, corpus_texts, embed_model, topk=5):
    q_emb = embed_model.encode([query_text], normalize_embeddings=True)[0]
    sims = np.dot(corpus_embeds, q_emb)
    top_idx = np.argsort(sims)[::-1][:topk]

    results = []
    for i in top_idx:
        results.append({
            "score": float(sims[i]),
            "text": corpus_texts[i]
        })
    return results, sims


def build_naive_rag_context(rag_results: List[dict]) -> str:
    blocks = []
    for i, r in enumerate(rag_results, 1):
        blocks.append(
            f"【相似病例 {i} | 相似度={r['score']:.3f}】\n{r['text']}"
        )
    return "\n\n".join(blocks)


# ======================================
# Prompt
# ======================================
SYSTEM_PROMPT = "你是一名经验丰富的中医辨证论治医生。"

USER_PROMPT_TEMPLATE = """
下面是当前病例的中医【判断线索】：
{features}

【相似病例参考】
{rag_context}

【中药使用约束】
你只能从下面给定的【中药全集】中选择药物：
{HERB_VOCAB_TEXT}

任务：
1. 生成一个合理的中药处方，给出10种中药；
2. 最后一行必须严格输出：
中药：药1, 药2, 药3, 药4, 药5, 药6, 药7, 药8, 药9, 药10
"""


# ======================================
# ======================================
def run_experiment(test_path, train_path, max_cases=100):
    p10_scores = []
    acc_cl_scores = []

    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)
    cache = load_category_cache(CATEGORY_CACHE_PATH, categories)
    herb_to_category.update(cache)

    fallback = (
        DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]
    )

    embed_model = SentenceTransformer("moka-ai/m3e-base")
    corpus_texts = load_naive_rag_corpus(train_path)
    corpus_embeds = embed_model.encode(corpus_texts, normalize_embeddings=True)

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_cases:
                break

            row = json.loads(line)
            structured = row["structured"]

            truth_text = structured.get("推荐中药", "")
            truth_herbs = [
                h for h in parse_truth_herbs(truth_text)
                if h and h not in ["无", "暂无", "无推荐", "无推荐中药"]
            ]
            if not truth_herbs:
                print(f"[DEBUG] 样本 {idx + 1} 无真实中药，跳过")
                continue

            structured_for_prompt = {k: v for k, v in structured.items() if k != "总结描述" and k != "推荐中药"}
            query_text = build_naive_query(structured_for_prompt)

            #print("\n" + "=" * 80)
            #print(query_text)

            rag_results, sims = naive_rag_search(
                query_text,
                corpus_embeds,
                corpus_texts,
                embed_model,
                TOPK_RAG
            )

            #print("max:", sims.max(), "mean:", sims.mean(), "min:", sims.min())

            #for i, r in enumerate(rag_results, 1):
                #print(f"\n--- Top {i} ---")
                #print(r["text"][:500])

            rag_context_text = build_naive_rag_context(rag_results)

            #print(rag_context_text[:1000])

            user_prompt = USER_PROMPT_TEMPLATE.format(
                features=json.dumps(structured_for_prompt, ensure_ascii=False, indent=2),
                rag_context=rag_context_text,
                HERB_VOCAB_TEXT=HERB_VOCAB_TEXT
            )

            #if idx < 2:
                #print(user_prompt)

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.001,
                "max_tokens": 1000
            }

            response = requests.post(
                GPT_API_URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=500
            )
            response.raise_for_status()
            result = response.json()

            raw_output = result["choices"][0]["message"]["content"].strip()
            pred_herbs = extract_herbs_from_output(raw_output, TOPK_HERB)

            p10 = p_at_k(pred_herbs, set(truth_herbs), TOPK_HERB)
            p10_scores.append(p10)

            pred_cats = herbs_to_categories(
                pred_herbs, herb_to_category, categories,
                CATEGORY_CACHE_PATH, fallback
            )
            truth_cats = herbs_to_categories(
                truth_herbs, herb_to_category, categories,
                CATEGORY_CACHE_PATH, fallback
            )

            p_cl, r_cl, f1_cl = f1_cl_score(pred_cats, truth_cats)

            acc_cl_scores.append(f1_cl)

            print("\n【ChatGPT 输出】")
            print(raw_output)
            print("预测 Top-10：", pred_herbs)
            print("真实中药：", truth_herbs)
            print("P@10 =", f"{p10:.3f}")
            print(f"CL-Precision = {p_cl:.3f}")
            print(f"CL-Recall    = {r_cl:.3f}")
            print(f"CL-F1        = {f1_cl:.3f}")

        print("\n====== naive RAG 最终结果 ======")
        if p10_scores:
            print(f"平均 P@10 = {float(np.mean(p10_scores)):.4f}")
        else:
            print("平均 P@10 = 0.0000")

        if acc_cl_scores:
            print(f"平均 Acc-CL@10 = {float(np.mean(acc_cl_scores)):.4f}")
        else:
            print("平均 Acc-CL@10 = 0.0000")

# ======================================
# ======================================
if __name__ == "__main__":
    run_experiment(
        test_path="test_jia_structured.jsonl",
        train_path="train_jia_structured.jsonl",
        max_cases=MAX_TEST_CASE
    )
