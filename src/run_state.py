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
from utils.hyperrag import rag_search, load_hyperedges, build_indexes

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
        "",
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
MAX_TEST_CASE = 100

HERB_VOCAB_PATH = SCRIPT_DIR / "unique_herbs.json"
EXPERIENCE_PATH = SCRIPT_DIR / "drug_association_top20.jsonl"
HYPEREDGES_PATH = SCRIPT_DIR / "hyperedges.jsonl"
TEST_PATH = SCRIPT_DIR / "test_jia_structured.jsonl"
TRAIN_PATH = SCRIPT_DIR / "train_jia_structured.jsonl"

GROUND_TRUTH_PATH = PROJECT_ROOT / "cases_jia.jsonl"
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
            meta = row.get("metadata", {})
            herb_text = meta.get("中药名称", "")
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


# ======================================
# ======================================
def load_experience_base(path):
    exps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if "status" not in e:
                e["status"] = "active"
            exps.append(e)

    if DEBUG:
        print("\n" + "=" * 60)
        print("[DEBUG] 经验库加载完成")
        print("[DEBUG] 经验条目数:", len(exps))
        if exps:
            print("[DEBUG] 第一条经验 keys:", list(exps[0].keys()))

    return exps


# ======================================
# ======================================
def build_rag_context_text(rag_results: List[dict]) -> str:
    if not rag_results:
        return "（无可参考的相似病例）"

    blocks = []
    for i, r in enumerate(rag_results, 1):
        edge = r["edge"]

        hit_entities_text = (
            "、".join(r["hit_entities"])
            if r["hit_entities"]
            else "（无明确结构命中，来自语义相似）"
        )

        blocks.append(
            f"【相似病例 {i} | edge_id={r['edge_id']} | 来源={'+'.join(r['sources'])}】\n"
            f"结构命中要素：{hit_entities_text}\n"
            f"综合相似度得分：{r['final_score']:.3f}\n"
            f"病例总结：{edge.get('总结描述', '')}\n"
            f"该病例所用中药：{edge.get('推荐中药', '')}"
        )

    return "\n\n".join(blocks)


# ======================================
# ======================================
EXPERIENCE_DISTILL_SYSTEM = "你是一名中医临床经验归纳助手，擅长从大规模经验库中提炼与当前病例最相关的经验要点。"

EXPERIENCE_DISTILL_PROMPT = """
下面是当前病例的中医【判断线索】（JSON）：
{features}

以下是候选【经验库条目】（多条）：
{experience_text}

任务：
1. 从经验库中提炼出与当前病例最相关的经验要点（可综合多条经验）；
2. 总结为一段自然语言，这里只讲：辨证思路 / 治则治法 / 推荐什么大类的药，如安神、化痰等，但不要出现具体的药物名；
3. 不要逐条复述经验条目，不要照抄原文；
4. 如果经验库中没有明显可用信息，输出：无可用经验

输出要求：
- 只输出一段经验总结文本（不超过 200 字）；
- 不要输出编号、列表、JSON。
"""


def distill_experience_by_llm(structured_features: dict, experiences: List[dict]) -> str:
    experience_text = "\n".join(
        json.dumps(e, ensure_ascii=False)
        for e in experiences
    )

    prompt = EXPERIENCE_DISTILL_PROMPT.format(
        features=json.dumps(structured_features, ensure_ascii=False, indent=2),
        experience_text=experience_text
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": EXPERIENCE_DISTILL_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 350
    }

    response = post_with_retry(payload, timeout=500)
    result = response.json()
    text = result["choices"][0]["message"]["content"].strip()

    if not text:
        return "无可用经验"

    return text


# ======================================
# ======================================
SYSTEM_PROMPT = "你是一名经验丰富的中医辨证论治医生。"

USER_PROMPT_TEMPLATE = """
下面是当前病例的中医【判断线索】（JSON）：

{features}

【过往经验总结】
{experience_summary}

【相似病例参考】
以下是与当前病例在不同判断子空间中相似的历史病例，
你在用药时可以重点加以参考。
{rag_context}

任务：
1. 结合中医【判断线索】和【相似病例参考】进行综合判断。
2. 生成一个合理的中药处方，给出10种中药；
3. 最后一行必须严格输出：
中药：药1, 药2, 药3, 药4, 药5, 药6, 药7, 药8, 药9, 药10
"""


# ======================================
# ======================================
def run_experiment(test_path, train_path, max_cases=100):
    p10_scores = []
    acc_cl_scores = []
    total_cases = 0
    skipped_cases = 0
    evaluated_cases = 0

    test_path = resolve_path(test_path, SCRIPT_DIR, PROJECT_ROOT)
    train_path = resolve_path(train_path, SCRIPT_DIR, PROJECT_ROOT)

    ground_truth = load_ground_truth_map(GROUND_TRUTH_PATH)
    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)
    cache = load_category_cache(CATEGORY_CACHE_PATH, categories)
    herb_to_category.update(cache)

    if not categories:
        raise RuntimeError("未发现中药大类目录，无法计算 Acc-CL@10")

    fallback_category = (
        DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]
    )

    all_gt_herbs = []
    for herbs in ground_truth.values():
        all_gt_herbs.extend(herbs)
    print("[进度] 开始加载真实值类别索引", flush=True)
    ensure_herbs_categorized(
        all_gt_herbs,
        herb_to_category,
        categories,
        CATEGORY_CACHE_PATH,
        fallback_category,
    )
    print("[进度] 真实值类别索引完成", flush=True)

    experiences = load_experience_base(EXPERIENCE_PATH)

    print("[进度] 开始加载经验库", flush=True)
    edges = load_hyperedges(str(HYPEREDGES_PATH))
    (
        embed_model,
        field_to_entities,
        field_to_index,
        summary_index
    ) = build_indexes(edges)
    print("[进度] 开始构建超图索引/向量", flush=True)

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_cases is not None and idx >= max_cases:
                break

            total_cases += 1
            row = json.loads(line)
            structured = row["structured"]

            case_id = str(row.get("id", "")).strip()
            truth_herbs = ground_truth.get(case_id, [])
            if not truth_herbs:
                if DEBUG:
                    print(f"[DEBUG] 样本 {idx + 1} 无有效真实中药，跳过评测")
                skipped_cases += 1
                continue
            evaluated_cases += 1

            structured_for_prompt = {
                k: v for k, v in structured.items()
                if k not in ["病位", "病性", "推荐中药"]
            }

            rag_results = rag_search(
                structured_query=structured,
                edges=edges,
                embed_model=embed_model,
                field_to_entities=field_to_entities,
                field_to_index=field_to_index,
                summary_index=summary_index,
                topk_entity=5,
                topk_summary=5,
                topk_per_field=2
            )

            rag_context_text = build_rag_context_text(rag_results)

            print("[进度] 开始 LLM 总结经验", flush=True)
            experience_summary = distill_experience_by_llm(
                structured_features=structured_for_prompt,
                experiences=experiences
            )
            print("[进度] LLM 总结经验完成", flush=True)

            if not experience_summary.strip():
                experience_summary = "无可用经验"

            user_prompt = USER_PROMPT_TEMPLATE.format(
                features=json.dumps(structured_for_prompt, ensure_ascii=False, indent=2),
                rag_context=rag_context_text,
                experience_summary=experience_summary
            )

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }

            print("[进度] 开始 LLM 生成推荐", flush=True)
            response = post_with_retry(payload, timeout=500)
            result = response.json()

            raw_output = result["choices"][0]["message"]["content"].strip()
            print("[进度] LLM 生成推荐完成", flush=True)
            pred_herbs = extract_herbs_from_output(raw_output, TOPK_HERB)

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

            print("\n【模型输出】")
            print(raw_output)
            print("\n预测 Top-10：", pred_herbs)
            print("真实中药：", truth_herbs)
            print(f"P@10 = {p10:.3f}")
            print("预测大类 Top-10：", pred_cats)
            print("真实大类：", sorted(set(truth_cats)))
            print(f"Acc-CL@10 = {f1_cl:.3f}")

        print(f"CL-Precision = {p_cl:.3f}")
        print(f"CL-Recall    = {r_cl:.3f}")
        print(f"CL-F1        = {f1_cl:.3f}")

    print("\n====== hypergraph 最终结果 ======")
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
    run_experiment(
        test_path=TEST_PATH,
        train_path=TRAIN_PATH,
        max_cases=MAX_TEST_CASE
    )
