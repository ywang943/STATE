# ======================================
# ======================================
import json
import time
import re
import os
import sys
import io
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

def _configure_console_encoding() -> None:
    # Avoid mojibake in Windows consoles.
    if os.name != "nt":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        return

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        output_cp = kernel32.GetConsoleOutputCP()
    except Exception:
        output_cp = None

    encoding = "utf-8" if output_cp == 65001 else "gb18030"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding=encoding, errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding=encoding, errors="replace")


_configure_console_encoding()

# ======================================
# ======================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"hyper_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


_orig_stdout = sys.stdout
_log_fp = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Tee(_orig_stdout, _log_fp)
print(f"[日志] 输出同步到: {LOG_FILE}")

# ======================================
# ======================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

REMOTE_API_KEY = os.getenv(
    "REMOTE_API_KEY",
    os.getenv(
        "GPT_API_KEY",
        "7bc70a62f11a48c18b00284cac02a7305753c7e4cce748bdb6d80791cfd32459",
    ),
)
GPT_API_URL = os.getenv(
    "GPT_API_URL",
    "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions",
).rstrip("/")
MODEL_NAME = os.getenv("REMOTE_MODEL", "gpt-4")

API_URL = GPT_API_URL
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": REMOTE_API_KEY,
}
print("使用的模型名称：", MODEL_NAME)

TOPK_HERB = 10
MAX_CASE = 100

HERB_VOCAB_PATH = SCRIPT_DIR / "unique_herbs.json"
EXPERIENCE_PATH = SCRIPT_DIR / "drug_association_top20_opt.jsonl"
HYPEREDGES_PATH = SCRIPT_DIR / "hyperedges.jsonl"
TEST_PATH = SCRIPT_DIR / "test_lung_structured_100.jsonl"
TRAIN_PATH = SCRIPT_DIR / "train_lung_structured.jsonl"

GROUND_TRUTH_PATH = PROJECT_ROOT / "cases_lung_structured.jsonl"
HERB_CATEGORY_DIR = PROJECT_ROOT / "herb_category"
CATEGORY_CACHE_PATH = PROJECT_ROOT / "herb_category_llm.json"
ENABLE_LLM_CATEGORY = True
CATEGORY_BATCH_SIZE = 20
DEFAULT_CATEGORY = "无分类"


# ======================================
# ======================================
def post_with_retry(payload, max_attempts=3, timeout=500, base_sleep=2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                API_URL,
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
DEBUG = True


# ======================================
# ======================================
with open(HERB_VOCAB_PATH, "r", encoding="utf-8") as f:
    HERB_VOCAB = json.load(f)

HERB_VOCAB_SET = set(HERB_VOCAB)


# ======================================
# ======================================
HERB_STOPWORDS = {"无", "暂无", "无推荐", "无推荐中药"}


def normalize_herb_name(name: str) -> str:
    return name.strip()


def resolve_path(path_value, *bases: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    for base in bases:
        candidate = base / path_value
        if candidate.exists():
            return candidate
    return path


def extract_herbs_from_output(text, topk=10):
    herbs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("中药："):
            content = line[len("中药：") :]
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


def load_ground_truth_map(input_path: Path):
    path = resolve_path(input_path, SCRIPT_DIR, PROJECT_ROOT)
    gt_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            case_id = str(row.get("id", "")).strip()
            #meta = row.get("metadata", {})
            structured = row.get("structured", {})
            herb_text = structured.get("推荐中药", "")

            herbs = [
                normalize_herb_name(h)
                for h in parse_truth_herbs(herb_text)
                if normalize_herb_name(h) and normalize_herb_name(h) not in HERB_STOPWORDS
            ]
            gt_map[case_id] = herbs
    return gt_map


def load_category_index(root_dir: Path):
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
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
            batch = pending[i : i + CATEGORY_BATCH_SIZE]
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
    herbs = [normalize_herb_name(h) for h in herbs if normalize_herb_name(h)]
    herb_to_category = ensure_herbs_categorized(
        herbs, herb_to_category, categories, cache_path, fallback_category
    )
    ordered = []
    for herb in herbs:
        cat = herb_to_category.get(herb, fallback_category)
        ordered.append(cat)
    return ordered

def build_structured_input(structured: dict):
    safe_keys = [
        "现病史",
        "中医四诊",
        "寒热",
        "虚实",
        "表里",
        "涉及可能脏腑",
        "涉及典型病机",
        "动态特征",
        "时间节律",
        "饮食相关",
        "情志相关",
        "消化表现"
    ]

    lines = []
    for k in safe_keys:
        v = structured.get(k, "")
        if v:
            lines.append(f"{k}: {v}")

    return "\n".join(lines)


def load_experience_base(path):
    exps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if "status" not in e:
                e["status"] = "active"
            exps.append(e)
    return exps


# ======================================
# ======================================
GENERATOR_SYSTEM = "你是一名精通中医辨证论治的医生助手。"

def llm_generator(structured_text, experiences):
    exp_text = "\n".join(
        f"[{e['id']}] {json.dumps(e, ensure_ascii=False)}"
        for e in experiences
    )

    prompt = f"""
以下是可参考的【经验库条目】，这些都基于医师的真实开方：
{exp_text}

【中医判断线索】
{structured_text}

任务：
1. 你可以使用【零条或多条经验】进行综合判断。
   - 如果使用经验，请在第一行注明：使用经验：[EXP_xxx, EXP_yyy, ...]
   - 如果未使用经验，请写：使用经验：无
   **注意：经验仅作为参考，不允许简单拼接或照抄经验内容**
2. 生成一个合理的中药处方，给出10种中药，数量必须是10；
3. 最后一行必须严格输出：
中药：药1,药2,药3,药4,药5,药6,药7,药8,药9,药10
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    response = requests.post(
        GPT_API_URL,
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    return result["choices"][0]["message"]["content"].strip()



def load_experience_base(path):
    exps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if "status" not in e:
                e["status"] = "active"
            exps.append(e)
    return exps



# ======================================
# ======================================
def llm_evaluator(pred_text, truth_herbs):
    pred_herbs = extract_herbs_from_output(pred_text, TOPK_HERB)
    hit = sum(1 for h in pred_herbs if h in truth_herbs)
    label = "positive" if hit >= 4 else "negative"
    return hit, label
# ======================================
# ======================================
def run_test_only():
    exps = load_experience_base(EXPERIENCE_PATH)
    p10_scores = []
    acc_cl_scores = []


    test_cases = []
    with open(PROJECT_ROOT / "test_lung_structured_100.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_cases.append(json.loads(line))

    if MAX_CASE:
        test_cases = test_cases[:MAX_CASE]

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

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    if MAX_CASE:
        cases = cases[:MAX_CASE]

    print(f"测试病例数：{len(cases)}")
    print(f"经验库条目数：{len(exps)}")

    for idx, case in enumerate(cases, 1):
        structured = case["structured"]
        truth_herbs = parse_truth_herbs(structured.get("推荐中药", ""))
        if not truth_herbs:
            continue

        structured_text = build_structured_input(structured)

        retrieved = exps

        gen = llm_generator(structured_text, retrieved)
        pred_herbs = extract_herbs_from_output(gen, TOPK_HERB)
        p10 = p_at_k(pred_herbs, set(truth_herbs), TOPK_HERB)
        p10_scores.append(p10)

        pred_cats = herbs_to_categories(
            pred_herbs,
            herb_to_category,
            categories,
            CATEGORY_CACHE_PATH,
            fallback_category,
        )
        truth_cats = herbs_to_categories(
            truth_herbs,
            herb_to_category,
            categories,
            CATEGORY_CACHE_PATH,
            fallback_category,
        )

        p_cl, r_cl, f1_cl = f1_cl_score(pred_cats, truth_cats)
        acc_cl_scores.append(f1_cl)
        print(f"Acc-CL@10 = {f1_cl:.3f}")

        hit, label = llm_evaluator(gen, truth_herbs)

        print("\n" + "=" * 60)
        print(f"CASE {idx}")
        print(gen)
        print(f"Hit = {hit}, Label = {label}, P@10 = {p10:.3f}")

    if p10_scores:
        print("\n" + "=" * 60)
        print(f"Test-only 平均 P@10 = {np.mean(p10_scores):.4f}")
    if acc_cl_scores:
        print(f"平均 Acc-CL@10 = {float(np.mean(acc_cl_scores)):.4f}")

if __name__ == "__main__":
    run_test_only()