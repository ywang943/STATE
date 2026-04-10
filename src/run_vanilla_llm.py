# ======================================
# ======================================
import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import requests

# ======================================
# ======================================
TOPK_HERB = 10
MAX_TEST_CASE = None

# ======================================
# ======================================
GPT_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
GPT_API_KEY = ""

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": GPT_API_KEY
}
MODEL_NAME = "gpt-4"

# ======================================
# ======================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

HERB_CATEGORY_DIR = PROJECT_ROOT / "herb_category"
CATEGORY_CACHE_PATH = PROJECT_ROOT / "herb_category_llm.json"
ENABLE_LLM_CATEGORY = True
CATEGORY_BATCH_SIZE = 20
DEFAULT_CATEGORY = "无分类"

HERB_STOPWORDS = {"无", "暂无", "无推荐", "无推荐中药"}

# ======================================
# ======================================
def post_with_retry(payload, max_attempts=3, timeout=500, base_sleep=2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                GPT_API_URL,
                headers=HEADERS,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"[WARN] 请求失败，{sleep_s}s 后重试：{exc}")
            time.sleep(sleep_s)
    raise last_error


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


# ======================================
# ======================================
def normalize_herb_name(name: str) -> str:
    return name.strip()

def extract_herbs_from_output(text, topk=10):
    """
    只从【以“中药：”开头】的那一行抽取
    """
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
    hit = sum(1 for h in pred[:k] if h in truth_set)
    return hit / k

# ======================================
# ======================================
def load_category_index(root_dir: Path):
    """
    扫描 herb_category/<大类>/*.txt
    返回：
      herb_to_category: dict[herb] -> category（若一药多类，取排序后第一个；与原逻辑一致）
      categories: list[str]
    """
    herb_to_categories = {}
    categories = set()

    root = Path(root_dir)
    if not root.exists():
        return {}, []

    for path in root.rglob("*.txt"):
        category = path.parent.name
        herb = normalize_herb_name(path.stem)
        categories.add(category)
        herb_to_categories.setdefault(herb, set()).add(category)

    herb_to_category = {
        herb: sorted(list(cats))[0] for herb, cats in herb_to_categories.items()
    }
    return herb_to_category, sorted(categories)

def load_category_cache(path: Path, categories: List[str]):
    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    cache = {}
    for herb, cats in raw.items():
        herb_norm = normalize_herb_name(herb)
        if isinstance(cats, list):
            candidates = cats
        else:
            candidates = [cats]
        valid = [c for c in candidates if c in categories]
        if valid:
            cache[herb_norm] = valid[0]
    return cache

def save_category_cache(path: Path, cache: dict):
    payload = {herb: [cat] for herb, cat in cache.items()}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def extract_json_object(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def choose_valid_category(value, categories: List[str]):
    if isinstance(value, list):
        for item in value:
            if item in categories:
                return item
        return None
    if isinstance(value, str) and value in categories:
        return value
    return None

def classify_herbs_via_llm(herbs, categories: List[str], fallback_category: str):
    """
    用 LLM 为缺失大类的 herb 分类；输出必须从 categories 中选 1 个。
    """
    if not herbs:
        return {}

    system = (
        "你是中药分类助手。请从给定的大类中为每味中药选择最合适的1个类别。"
        "只能从大类列表中选，且只能选1类。仅输出JSON对象，不要额外文字。"
    )
    category_text = "、".join(categories)

    results = {}
    pending = [normalize_herb_name(h) for h in herbs if normalize_herb_name(h)]

    for attempt in range(2):
        if not pending:
            break

        new_results = {}
        for i in range(0, len(pending), CATEGORY_BATCH_SIZE):
            batch = pending[i:i + CATEGORY_BATCH_SIZE]
            herb_text = "\n".join(f"- {h}" for h in batch)

            user = (
                f"大类列表：{category_text}\n"
                "请为下面每味中药选择一个类别（只能从大类列表中选）：\n"
                f"{herb_text}\n"
                "输出格式示例：\n"
                '{"中药A": "大类1", "中药B": "大类2"}'
            )

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": 600,
            }

            print("[进度] 开始分类中药批次", flush=True)
            response = post_with_retry(payload, timeout=500)
            result = response.json()
            raw_output = result["choices"][0]["message"]["content"].strip()
            print("[进度] 中药批次分类完成", flush=True)

            data = None
            try:
                data = json.loads(raw_output)
            except json.JSONDecodeError:
                data = extract_json_object(raw_output)

            if not isinstance(data, dict):
                continue

            for herb, cat in data.items():
                herb_norm = normalize_herb_name(str(herb))
                valid = choose_valid_category(cat, categories)
                if valid:
                    new_results[herb_norm] = valid

        results.update(new_results)
        pending = [h for h in pending if h not in results]

    for herb in pending:
        results[herb] = fallback_category

    return results

def ensure_herbs_categorized(
    herbs, herb_to_category, categories, cache_path, fallback_category
):
    """
    对输入 herbs 做缺失补全：
    - 若 herb 不在 herb_to_category：用 LLM 分类/或 fallback
    - 并写回 cache
    """
    missing = [
        normalize_herb_name(h)
        for h in herbs
        if normalize_herb_name(h) and normalize_herb_name(h) not in herb_to_category
    ]
    if not missing:
        return herb_to_category

    if not ENABLE_LLM_CATEGORY:
        for herb in missing:
            herb_to_category[herb] = fallback_category
        return herb_to_category

    new_map = classify_herbs_via_llm(missing, categories, fallback_category)
    for herb, cat in new_map.items():
        herb_to_category[herb] = cat

    cache = load_category_cache(cache_path, categories)
    for herb, cat in new_map.items():
        cache[herb] = cat
    save_category_cache(cache_path, cache)

    return herb_to_category

def herbs_to_categories(
    herbs, herb_to_category, categories, cache_path, fallback_category
):
    """
    herb 列表 -> category 列表（保持顺序），并确保缺失的 herb 会先被分类补全
    """
    herbs = [normalize_herb_name(h) for h in herbs if normalize_herb_name(h)]
    herb_to_category = ensure_herbs_categorized(
        herbs, herb_to_category, categories, cache_path, fallback_category
    )
    ordered = []
    for herb in herbs:
        cat = herb_to_category.get(herb, fallback_category)
        ordered.append(cat)
    return ordered

# ======================================
# ======================================
SYSTEM_PROMPT = "你是一名精通中医辨证论治的医生助手。"

USER_QUERY = (
    "下面给出了某个病例的现病史、中医四诊以及【中医判断线索】，这是从病史和四诊中提取出的结构化观察结果。\n"
    "请你基于这些内容，给出一个合理的中药用药组合。\n"
    "可以简要说明用药思路。\n\n"
    "要求：\n"
    "1. 最后一行必须给出【10味】中药\n"
    "2. 格式必须严格为：\n"
    "中药：药1, 药2, 药3, 药4, 药5, 药6, 药7, 药8, 药9, 药10\n"
)

# ======================================
# ======================================
with open(PROJECT_ROOT / "unique_herbs.txt", "r", encoding="utf-8") as f:
    herb_vocab = [line.strip() for line in f if line.strip()]

HERB_VOCAB_TEXT = "，".join(herb_vocab)

HERB_CONSTRAINT_PROMPT = (
    "【中药使用约束】\n"
    "你只能从下面给定的【中药全集】中选择药物，\n"
    "不得使用列表之外的任何药名（包括别名、炮制名、合成名）。\n\n"
    f"中药全集：\n{HERB_VOCAB_TEXT}\n"
)

# ======================================
# ======================================
def run_experiment(input_path: str):
    test_cases = []
    with open(PROJECT_ROOT / input_path, "r", encoding="utf-8") as f:
        for line in f:
            test_cases.append(json.loads(line))

    if MAX_TEST_CASE:
        test_cases = test_cases[:MAX_TEST_CASE]

    print(f"测试病例数：{len(test_cases)}")

    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)

    if not categories:
        raise RuntimeError("未发现中药大类目录，无法计算 Acc-CL@10")

    fallback_category = (
        DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]
    )

    cache = load_category_cache(CATEGORY_CACHE_PATH, categories)
    herb_to_category.update(cache)

    all_gt_herbs = []
    for row in test_cases:
        truth_text = row.get("structured", {}).get("推荐中药", "")
        truth_herbs = [
            normalize_herb_name(h)
            for h in parse_truth_herbs(truth_text)
            if normalize_herb_name(h) and normalize_herb_name(h) not in HERB_STOPWORDS
        ]
        all_gt_herbs.extend(truth_herbs)

    if all_gt_herbs:
        print("[进度] 开始加载真实值类别索引", flush=True)
        ensure_herbs_categorized(
            all_gt_herbs,
            herb_to_category,
            categories,
            CATEGORY_CACHE_PATH,
            fallback_category,
        )
        print("[进度] 真实值类别索引完成", flush=True)

    p10_scores = []
    acc_cl_scores = []
    total_cases = 0
    skipped_cases = 0
    evaluated_cases = 0

    for idx, row in enumerate(test_cases, 1):
        total_cases += 1
        structured = row["structured"]

        # ---------- Ground Truth ----------
        truth_text = structured.get("推荐中药", "")
        truth_herbs = [
            normalize_herb_name(h)
            for h in parse_truth_herbs(truth_text)
            if normalize_herb_name(h) and normalize_herb_name(h) not in HERB_STOPWORDS
        ]

        if not truth_herbs:
            skipped_cases += 1
            continue

        evaluated_cases += 1
        truth_set = set(truth_herbs)

        structured_text = (
            f"现病史: {structured.get('现病史', '')}\n"
            f"中医四诊: {structured.get('中医四诊', '')}\n"
            f"寒热: {structured.get('寒热','')}\n"
            f"虚实: {structured.get('虚实','')}\n"
            f"表里: {structured.get('表里','')}\n"
            f"涉及可能脏腑: {structured.get('涉及可能脏腑','')}\n"
            f"涉及典型病机: {structured.get('涉及典型病机','')}\n"
            f"动态特征: {structured.get('动态特征','')}\n"
            f"时间节律: {structured.get('时间节律','')}\n"
            f"饮食相关: {structured.get('饮食相关','')}\n"
            f"情志相关: {structured.get('情志相关','')}\n"
            f"消化表现: {structured.get('消化表现','')}\n"
        )

        prompt = (
            f"【中医判断线索】\n"
            f"{structured_text}\n"
            f"{HERB_CONSTRAINT_PROMPT}\n"
            f"{USER_QUERY}"
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        response = post_with_retry(payload, timeout=500)
        result = response.json()
        raw_output = result["choices"][0]["message"]["content"].strip()
        pred_herbs = extract_herbs_from_output(raw_output, TOPK_HERB)

        # ---------- P@10 ----------
        p10 = p_at_k(pred_herbs, truth_set, TOPK_HERB)
        p10_scores.append(p10)

        pred_categories = herbs_to_categories(
            pred_herbs,
            herb_to_category,
            categories,
            CATEGORY_CACHE_PATH,
            fallback_category,
        )
        truth_categories = herbs_to_categories(
            truth_herbs,
            herb_to_category,
            categories,
            CATEGORY_CACHE_PATH,
            fallback_category,
        )
        p_cl, r_cl, f1_cl = f1_cl_score(pred_categories, truth_categories)

        acc_cl_scores.append(f1_cl)

        print("\n" + "=" * 60)
        print(f"样本 {idx}")
        print("模型输出：")
        print(raw_output)
        print("\n预测 Top-10：", pred_herbs)
        print("真实中药：", truth_herbs)
        print(f"P@10 = {p10:.3f}")
        print("预测大类 Top-10：", pred_categories)
        print("真实大类：", sorted(set(truth_categories)))
        print(f"CL-Precision = {p_cl:.3f}")
        print(f"CL-Recall    = {r_cl:.3f}")
        print(f"CL-F1        = {f1_cl:.3f}")

    print("\n" + "=" * 60)
    if p10_scores:
        print(f"平均 P@10 = {float(np.mean(p10_scores)):.4f}")
    else:
        print("平均 P@10 = 0.0000")

    if acc_cl_scores:
        print(f"平均 Acc-CL@10 = {float(np.mean(acc_cl_scores)):.4f}")
    else:
        print("平均 Acc-CL@10 = 0.0000")

    print("\n====== 有效样本统计 ======")
    print(f"有效样本数 = {evaluated_cases} / {total_cases} (跳过 {skipped_cases})")

# ======================================
# ======================================
if __name__ == "__main__":
    run_experiment("test_jia_structured.jsonl")
