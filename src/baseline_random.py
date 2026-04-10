# ======================================
# ======================================
import json
import re
import random
from typing import List
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

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

# ======================================
# ======================================
TOPK_HERB = 10
TOPK_RAG = 5
MAX_TEST_CASE = 100

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
            parts = re.split(r"[，,、\s]+", line[len("中药："):])
            herbs = [p.strip() for p in parts if p.strip()]
            break
    herbs = herbs[:topk] + [""] * max(0, topk - len(herbs))
    return herbs


def parse_truth_herbs(text):
    return [h.strip() for h in re.split(r"[，,、\s]+", text) if h.strip()]


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
    P, G = set(pred_categories), set(truth_categories)
    if not P or not G:
        return 0.0, 0.0, 0.0
    inter = len(P & G)
    precision = inter / len(P)
    recall = inter / len(G)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
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
            obj = json.loads(re.search(r"\{.*\}", data["choices"][0]["message"]["content"], re.S).group())
            cat = list(obj.values())[0]
            results[herb] = cat if cat in categories else fallback
        except Exception:
            results[herb] = fallback
    return results


def ensure_herbs_categorized(herbs, herb_to_category, categories, cache_path, fallback):
    missing = [h for h in herbs if h and h not in herb_to_category]
    if not missing:
        return herb_to_category

    new_map = classify_herbs_via_llm(missing, categories, fallback) \
        if ENABLE_LLM_CATEGORY else {h: fallback for h in missing}

    herb_to_category.update(new_map)
    cache = load_category_cache(cache_path, categories)
    cache.update(new_map)
    save_category_cache(cache_path, cache)
    return herb_to_category


def herbs_to_categories(herbs, herb_to_category, categories, cache_path, fallback):
    herbs = [normalize_herb_name(h) for h in herbs if h]
    herb_to_category = ensure_herbs_categorized(
        herbs, herb_to_category, categories, cache_path, fallback
    )
    return [herb_to_category.get(h, fallback) for h in herbs]

# ======================================
# ======================================
def build_naive_query(structured: dict) -> str:
    return "\n".join(f"{k}：{v}" for k, v in structured.items())

def load_naive_rag_corpus(train_path):
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(build_naive_query(row["structured"]))
    return texts

def random_rag_search(corpus_texts, topk=5):
    samples = random.sample(corpus_texts, topk)
    return [{"score": None, "text": t} for t in samples]

def build_rag_context(rag_results, mode):
    blocks = []
    for i, r in enumerate(rag_results, 1):
        if mode == "naive":
            blocks.append(f"【相似病例 {i} | 相似度={r['score']:.3f}】\n{r['text']}")
        else:
            blocks.append(f"【随机参考病例 {i}】\n{r['text']}")
    return "\n\n".join(blocks)

# ======================================
# Prompt
# ======================================
SYSTEM_PROMPT = "你是一名经验丰富的中医辨证论治医生。"

USER_PROMPT_TEMPLATE = """
下面是当前病例的中医【判断线索】：
{features}

【病例参考】
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
def run_experiment(test_path, train_path, rag_mode="random", max_cases=100):
    print(f"\n[INFO] RAG 模式：{rag_mode}")

    p10_scores, f1_scores = [], []

    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)
    herb_to_category.update(load_category_cache(CATEGORY_CACHE_PATH, categories))
    fallback = DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]

    corpus_texts = load_naive_rag_corpus(train_path)


    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_cases:
                break

            row = json.loads(line)
            structured = row["structured"]

            truth_herbs = [
                h for h in parse_truth_herbs(structured.get("推荐中药", ""))
                if h not in HERB_STOPWORDS
            ]
            if not truth_herbs:
                continue

            query = build_naive_query({"现病史": structured.get("现病史", "")})

            rag_results = random_rag_search(corpus_texts, TOPK_RAG)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                features=json.dumps({"现病史": structured.get("现病史", "")}, ensure_ascii=False, indent=2),
                rag_context=build_rag_context(rag_results, rag_mode),
                HERB_VOCAB_TEXT=HERB_VOCAB_TEXT
            )

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.001,
                "max_tokens": 1000
            }

            r = requests.post(GPT_API_URL, headers=HEADERS, data=json.dumps(payload), timeout=500)
            result = r.json()

            pred_herbs = extract_herbs_from_output(result["choices"][0]["message"]["content"])
            p10_scores.append(p_at_k(pred_herbs, set(truth_herbs), TOPK_HERB))

            pred_cats = herbs_to_categories(pred_herbs, herb_to_category, categories, CATEGORY_CACHE_PATH, fallback)
            truth_cats = herbs_to_categories(truth_herbs, herb_to_category, categories, CATEGORY_CACHE_PATH, fallback)
            _, _, f1 = f1_cl_score(pred_cats, truth_cats)
            f1_scores.append(f1)

            print(f"\n样本 {idx+1} | RAG={rag_mode}")
            print("P@10 =", f"{p10_scores[-1]:.3f}", "CL-F1 =", f"{f1:.3f}")

    print("\n====== 最终结果 ======")
    print("平均 P@10 =", f"{np.mean(p10_scores):.4f}")
    print("平均 CL-F1 =", f"{np.mean(f1_scores):.4f}")

# ======================================
# ======================================
if __name__ == "__main__":
    #run_experiment("test_jia_structured.jsonl", "train_jia_structured.jsonl", rag_mode="naive")
    run_experiment("test_jia_structured.jsonl", "train_jia_structured.jsonl", rag_mode="random")
