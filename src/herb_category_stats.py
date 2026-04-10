# ======================================
# ======================================
import json
import re
import os
import time
from pathlib import Path
from collections import Counter
import requests

# ======================================
# ======================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

DATASET_PATH = PROJECT_ROOT / "cases_jia_structured.jsonl"
HERB_CATEGORY_DIR = PROJECT_ROOT / "herb_category"
CATEGORY_CACHE_PATH = PROJECT_ROOT / "herb_category_llm.json"

DEFAULT_CATEGORY = "无分类"
ENABLE_LLM_CATEGORY = True
CATEGORY_BATCH_SIZE = 20

# ======================================
# ======================================
REMOTE_API_KEY = os.getenv(
    "REMOTE_API_KEY",
    os.getenv(
        "GPT_API_KEY",
        "",
    ),
)

GPT_API_URL = os.getenv(
    "GPT_API_URL",
    "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions",
).rstrip("/")

MODEL_NAME = os.getenv("REMOTE_MODEL", "gpt-4")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": REMOTE_API_KEY,
}

# ======================================
# ======================================
def post_with_retry(payload, max_attempts=3, timeout=500, base_sleep=2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                GPT_API_URL,
                headers=HEADERS,
                data=json.dumps(payload),
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

# ======================================
# ======================================
def normalize_herb_name(name: str) -> str:
    return name.strip()

def parse_herbs(text: str):
    return [
        h.strip()
        for h in re.split(r"[，,、\s]+", text)
        if h.strip()
    ]

def extract_json_object(text: str):
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def choose_valid_category(value, categories):
    if isinstance(value, list):
        for item in value:
            if item in categories:
                return item
    if isinstance(value, str) and value in categories:
        return value
    return None

# ======================================
# ======================================
def load_category_index(root_dir: Path):
    herb_to_categories = {}
    categories = set()

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

def load_category_cache(path: Path, categories):
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    cache = {}
    for herb, cat in raw.items():
        if cat in categories:
            cache[normalize_herb_name(herb)] = cat
    return cache

def save_category_cache(path: Path, cache: dict):
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# ======================================
# ======================================
def classify_herbs_via_llm(herbs, categories, fallback_category):
    if not herbs:
        return {}

    system = (
        "你是中药分类助手。请从给定的大类中为每味中药选择最合适的1个类别。"
        "只能从大类列表中选，且只能选1类。仅输出JSON对象，不要额外文字。"
    )
    category_text = "、".join(categories)

    results = {}
    pending = [normalize_herb_name(h) for h in herbs if normalize_herb_name(h)]

    for _ in range(2):
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

            response = post_with_retry(payload)
            raw = response.json()["choices"][0]["message"]["content"].strip()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = extract_json_object(raw)

            if not isinstance(data, dict):
                continue

            for herb, cat in data.items():
                herb_norm = normalize_herb_name(herb)
                valid = choose_valid_category(cat, categories)
                if valid:
                    new_results[herb_norm] = valid

        results.update(new_results)
        pending = [h for h in pending if h not in results]

    for h in pending:
        results[h] = fallback_category

    return results

def ensure_herbs_categorized(
    herbs, herb_to_category, categories, cache_path, fallback_category
):
    missing = [h for h in herbs if h not in herb_to_category]

    if not missing:
        return herb_to_category

    if not ENABLE_LLM_CATEGORY:
        for h in missing:
            herb_to_category[h] = fallback_category
        return herb_to_category

    new_map = classify_herbs_via_llm(missing, categories, fallback_category)
    herb_to_category.update(new_map)

    cache = load_category_cache(cache_path, categories)
    cache.update(new_map)
    save_category_cache(cache_path, cache)

    return herb_to_category

# ======================================
# ======================================
def main():
    print("[1] 读取数据集")
    herbs = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("structured", {}).get("推荐中药", "")
            herbs.extend(parse_herbs(text))

    herbs = [normalize_herb_name(h) for h in herbs if h]
    herb_counter = Counter(herbs)
    unique_herbs = sorted(herb_counter.keys())

    print(f"    共发现中药 {len(unique_herbs)} 种")

    print("[2] 加载已有大类体系")
    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)
    cache = load_category_cache(CATEGORY_CACHE_PATH, categories)
    herb_to_category.update(cache)

    fallback = (
        DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]
    )

    print("[3] LLM 补全缺失分类")
    herb_to_category = ensure_herbs_categorized(
        unique_herbs,
        herb_to_category,
        categories,
        CATEGORY_CACHE_PATH,
        fallback
    )

    out_path = PROJECT_ROOT / "herb_to_category_full.json"
    out_path.write_text(
        json.dumps(herb_to_category, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n✅ 完成，结果已保存：{out_path}")

    print("\n示例：")
    for h in unique_herbs[:10]:
        print(f"{h} -> {herb_to_category[h]}")

if __name__ == "__main__":
    main()
