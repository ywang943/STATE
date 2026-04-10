# ======================================
#
# ======================================

import json
import time
import re
import os
import sys
import io
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ======================================
# ======================================
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ======================================
# ======================================
def _configure_console_encoding() -> None:
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
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"mdagents_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
REMOTE_API_KEY = os.getenv(
    "REMOTE_API_KEY",
    os.getenv("GPT_API_KEY",
              "7bc70a62f11a48c18b00284cac02a7305753c7e4cce748bdb6d80791cfd32459"),
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
print(f"[配置] 模型：{MODEL_NAME}")

# ======================================
# ======================================
TOPK_HERB           = 10
MAX_TEST_CASE       = 100
RAG_TOPK            = 3
DEBATE_ROUNDS       = 1
DEBUG               = True

# ======================================
# ======================================
TRAIN_PATH          = SCRIPT_DIR / "train_wei.jsonl"
TEST_PATH           = SCRIPT_DIR / "test_wei.jsonl"
GROUND_TRUTH_PATH   = SCRIPT_DIR / "cases_wei.jsonl"
HERB_CATEGORY_DIR   = SCRIPT_DIR / "herb_category"
CATEGORY_CACHE_PATH = SCRIPT_DIR / "herb_category_llm.json"

ENABLE_LLM_CATEGORY = True
CATEGORY_BATCH_SIZE = 20
DEFAULT_CATEGORY    = "无分类"


# ======================================
# ======================================
def post_with_retry(payload, max_attempts=3, timeout=500, base_sleep=2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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


def call_llm(system: str, user: str, temperature: float = 0.2,
             max_tokens: int = 600) -> str:
    """统一 LLM 调用入口，返回文本"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = post_with_retry(payload)
    return resp.json()["choices"][0]["message"]["content"].strip()


# ======================================
# ======================================
HERB_STOPWORDS = {"无", "暂无", "无推荐", "无推荐中药"}

def normalize_herb_name(name: str) -> str:
    return name.strip()

def extract_herbs_from_output(text: str, topk: int = 10) -> List[str]:
    """从输出文本中提取最后一行 '中药：...' 格式的药物列表"""
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

def parse_truth_herbs(text: str) -> List[str]:
    herbs = re.split(r"[，,、\s]+", text)
    return [h.strip() for h in herbs if h.strip()]

def p_at_k(pred: List[str], truth_set: set, k: int = 10) -> float:
    return sum(1 for h in pred[:k] if h in truth_set) / k

def f1_cl_score(pred_cats, truth_cats) -> Tuple[float, float, float]:
    P, G = set(pred_cats), set(truth_cats)
    if not P or not G:
        return 0.0, 0.0, 0.0
    inter = len(P & G)
    precision = inter / len(P)
    recall    = inter / len(G)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def load_ground_truth_map(input_path: Path) -> Dict[str, List[str]]:
    gt_map = {}
    with open(input_path, "r", encoding="utf-8") as f:
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
    herb_to_categories: Dict[str, set] = {}
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

def load_category_cache(path: Path, categories: List[str]) -> dict:
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
        candidates = cats if isinstance(cats, list) else [cats]
        valid = [c for c in candidates if c in categories]
        if valid:
            cache[herb_norm] = valid[0]
    return cache

def save_category_cache(path: Path, cache: dict):
    payload = {herb: [cat] for herb, cat in cache.items()}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def extract_json_object(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def choose_valid_category(value, categories: List[str]) -> Optional[str]:
    if isinstance(value, list):
        for item in value:
            if item in categories:
                return item
        return None
    if isinstance(value, str) and value in categories:
        return value
    return None

def classify_herbs_via_llm(herbs: List[str], categories: List[str],
                            fallback_category: str) -> dict:
    if not herbs:
        return {}
    system = (
        "你是中药分类助手。请从给定的大类中为每味中药选择最合适的1个类别。"
        "只能从大类列表中选，且只能选1类。仅输出JSON对象，不要额外文字。"
    )
    category_text = "、".join(categories)
    results = {}
    pending = [normalize_herb_name(h) for h in herbs if normalize_herb_name(h)]

    for _attempt in range(2):
        if not pending:
            break
        new_results = {}
        for i in range(0, len(pending), CATEGORY_BATCH_SIZE):
            batch = pending[i: i + CATEGORY_BATCH_SIZE]
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
            resp = post_with_retry(payload, timeout=500)
            raw_output = resp.json()["choices"][0]["message"]["content"].strip()
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

def ensure_herbs_categorized(herbs, herb_to_category, categories,
                              cache_path, fallback_category):
    missing = [
        normalize_herb_name(h) for h in herbs
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

def herbs_to_categories(herbs, herb_to_category, categories,
                         cache_path, fallback_category) -> List[str]:
    herbs = [normalize_herb_name(h) for h in herbs if normalize_herb_name(h)]
    herb_to_category = ensure_herbs_categorized(
        herbs, herb_to_category, categories, cache_path, fallback_category
    )
    return [herb_to_category.get(h, fallback_category) for h in herbs]


# ======================================
# ======================================
class SimpleRAG:
    """
    sentence-transformers 余弦相似度检索，仅用 train 集构建索引。
    """
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"[RAG] 加载 embedding 模型：{model_name}")
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(model_name)
        self.train_cases: List[dict] = []
        self.train_embeddings: Optional[np.ndarray] = None

    @staticmethod
    def _case_to_text(structured: dict) -> str:
        fields = ["现病史", "四诊(规范)", "中医辨证", "中医诊断",
                  "西医诊断", "病性(泛化)", "病位(泛化)"]
        parts = [f"{f}：{structured[f]}" for f in fields if structured.get(f)]
        return " ".join(parts)

    def build_index(self, train_path: Path):
        print("[RAG] 构建 train 索引...")
        cases = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                cases.append(json.loads(line))
        self.train_cases = cases
        texts = [self._case_to_text(r.get("structured", {})) for r in cases]
        self.train_embeddings = self.embed_model.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        print(f"[RAG] 索引构建完成，共 {len(cases)} 条训练病例")

    def retrieve(self, structured: dict, topk: int = RAG_TOPK) -> List[dict]:
        if self.train_embeddings is None:
            raise RuntimeError("请先调用 build_index()")
        query_emb = self.embed_model.encode(
            [self._case_to_text(structured)], normalize_embeddings=True
        )[0]
        scores = self.train_embeddings @ query_emb
        topk_idx = np.argsort(scores)[::-1][:topk]
        results = []
        for i in topk_idx:
            s = self.train_cases[i].get("structured", {})
            results.append({
                "score":    float(scores[i]),
                "现病史":   s.get("现病史", ""),
                "四诊":     s.get("四诊(规范)", ""),
                "中医辨证": s.get("中医辨证", ""),
                "中医诊断": s.get("中医诊断", ""),
                "病性":     s.get("病性(泛化)", ""),
                "病位":     s.get("病位(泛化)", ""),
                "推荐中药": s.get("推荐中药", ""),
            })
        return results


def build_rag_context(similar_cases: List[dict]) -> str:
    if not similar_cases:
        return "（无可参考的相似病例）"
    blocks = []
    for i, c in enumerate(similar_cases, 1):
        blocks.append(
            f"【相似病例 {i}】相似度={c['score']:.3f}\n"
            f"  现病史：{c['现病史']}\n"
            f"  四诊：{c['四诊']}\n"
            f"  中医辨证：{c['中医辨证']}\n"
            f"  病性：{c['病性']}  病位：{c['病位']}\n"
            f"  所用中药：{c['推荐中药']}"
        )
    return "\n\n".join(blocks)


# ======================================
# ======================================

WANGZHEN_SYSTEM = (
    "你是一名擅长望、闻、问、切的中医望诊医师，"
    "擅长从患者的症状描述、舌象、脉象等四诊信息中提取关键辨证线索。"
)

WANGZHEN_PROMPT = """\
【当前病例结构化信息】
{features}

【相似病例参考（来自历史病案库）】
{rag_context}

请你作为望诊医师完成以下任务：
1. 归纳该患者最重要的症状和四诊特征（3-5条）；
2. 根据这些特征初步判断主要病机；
3. 推荐你认为最关键的 5 味核心中药（只写药名，逗号分隔）。

输出格式（严格遵守）：
关键症状：...
初步病机：...
推荐核心药物：药1, 药2, 药3, 药4, 药5\
"""

BIANZHEN_SYSTEM = (
    "你是一名专注于辨证论治的中医辨证医师，"
    "擅长从病性、病位角度分析疾病本质，制定治则治法。"
)

BIANZHEN_PROMPT = """\
【当前病例结构化信息】
{features}

【相似病例参考（来自历史病案库）】
{rag_context}

【望诊医师意见】
{wangzhen_opinion}

请你作为辨证医师完成以下任务：
1. 结合病性（{bing_xing}）和病位（{bing_wei}），深化辨证分析；
2. 明确治则治法（如滋阴养血、疏肝解郁等）；
3. 根据治法推荐最关键的 5 味核心中药（可与望诊医师有不同意见）。

输出格式（严格遵守）：
辨证分析：...
治则治法：...
推荐核心药物：药1, 药2, 药3, 药4, 药5\
"""

DEBATE_PROMPT = """\
【辩论第 {round}/{total} 轮】

病例信息：
{features}

【望诊医师当前意见】
{wangzhen_opinion}

【你（辨证医师）上一轮意见】
{bianzhen_opinion}

请你进一步思考：
- 你是否同意望诊医师的判断？若有分歧，请明确说明理由；
- 在综合考虑后，给出你调整后的 5 味核心药物推荐。

输出格式（严格遵守）：
回应与补充：...
调整后推荐核心药物：药1, 药2, 药3, 药4, 药5\
"""

CHUFANG_SYSTEM = (
    "你是一名资深中医处方医师，擅长将多位医师的辨证意见转化为完整合理的候选药方。"
)

CHUFANG_PROMPT = """\
【当前病例结构化信息】
{features}

【相似病例参考（来自历史病案库）】
{rag_context}

【望诊医师意见】
{wangzhen_opinion}

【辨证医师意见】
{bianzhen_opinion}

请你作为处方医师完成以下任务：
1. 找出两位医师的共识药物，作为处方核心；
2. 针对分歧之处，结合相似病例和你的临床经验作出取舍；
3. 补充君臣佐使，拟定一份包含 10 味中药的候选处方；
4. 简要说明组方逻辑。

输出格式（严格遵守）：
组方逻辑：...
候选处方：药1, 药2, 药3, 药4, 药5, 药6, 药7, 药8, 药9, 药10\
"""

ZHUREN_SYSTEM = (
    "你是一名权威中医主任医师，负责对会诊意见进行最终裁定，"
    "给出唯一确定的最终处方。你的决定不可更改。"
)

ZHUREN_PROMPT = """\
【当前病例结构化信息】
{features}

【相似病例参考（来自历史病案库）】
{rag_context}

以下是本次会诊各医师意见：

【望诊医师意见】
{wangzhen_opinion}

【辨证医师意见】
{bianzhen_opinion}

【处方医师候选处方】
{chufang_opinion}

请你作为主任医师进行最终裁定：
1. 评估各医师意见的合理性，点出关键依据；
2. 对候选处方进行调整（如有必要）；
3. 确定最终处方（恰好 10 味中药）。

最后一行必须严格按以下格式输出，不得有任何变动：
中药：药1, 药2, 药3, 药4, 药5, 药6, 药7, 药8, 药9, 药10\
"""


# ======================================
# ======================================
def run_mdagents(structured: dict, rag_context: str) -> str:
    """
    四角色多智能体协作：
      望诊医师 → 辨证医师（含可选辩论）→ 处方医师 → 主任医师裁定
    """
    features = json.dumps(
        {k: v for k, v in structured.items() if k != "推荐中药"},
        ensure_ascii=False, indent=2
    )
    bing_xing = structured.get("病性(泛化)", "")
    bing_wei  = structured.get("病位(泛化)", "")

    print("[MDAgents] Step1 望诊医师分析...")
    wangzhen_opinion = call_llm(
        WANGZHEN_SYSTEM,
        WANGZHEN_PROMPT.format(features=features, rag_context=rag_context),
        temperature=0.3, max_tokens=400,
    )
    if DEBUG:
        print("  [望诊医师]\n", wangzhen_opinion)

    print("[MDAgents] Step2 辨证医师分析...")
    bianzhen_opinion = call_llm(
        BIANZHEN_SYSTEM,
        BIANZHEN_PROMPT.format(
            features=features,
            rag_context=rag_context,
            wangzhen_opinion=wangzhen_opinion,
            bing_xing=bing_xing,
            bing_wei=bing_wei,
        ),
        temperature=0.3, max_tokens=400,
    )
    if DEBUG:
        print("  [辨证医师]\n", bianzhen_opinion)

    for r in range(1, DEBATE_ROUNDS + 1):
        print(f"[MDAgents] 辩论第 {r}/{DEBATE_ROUNDS} 轮...")
        bianzhen_opinion = call_llm(
            BIANZHEN_SYSTEM,
            DEBATE_PROMPT.format(
                round=r, total=DEBATE_ROUNDS,
                features=features,
                wangzhen_opinion=wangzhen_opinion,
                bianzhen_opinion=bianzhen_opinion,
            ),
            temperature=0.3, max_tokens=350,
        )
        if DEBUG:
            print(f"  [辩论轮{r} 辨证医师]\n", bianzhen_opinion)

    print("[MDAgents] Step4 处方医师拟方...")
    chufang_opinion = call_llm(
        CHUFANG_SYSTEM,
        CHUFANG_PROMPT.format(
            features=features,
            rag_context=rag_context,
            wangzhen_opinion=wangzhen_opinion,
            bianzhen_opinion=bianzhen_opinion,
        ),
        temperature=0.2, max_tokens=500,
    )
    if DEBUG:
        print("  [处方医师]\n", chufang_opinion)

    print("[MDAgents] Step5 主任医师最终裁定...")
    final_output = call_llm(
        ZHUREN_SYSTEM,
        ZHUREN_PROMPT.format(
            features=features,
            rag_context=rag_context,
            wangzhen_opinion=wangzhen_opinion,
            bianzhen_opinion=bianzhen_opinion,
            chufang_opinion=chufang_opinion,
        ),
        temperature=0.1, max_tokens=600,
    )
    if DEBUG:
        print("  [主任医师最终]\n", final_output)

    return final_output


# ======================================
# ======================================
def run_experiment(test_path: Path, train_path: Path, max_cases: int = 100):
    p10_scores    = []
    acc_cl_scores = []
    total_cases   = 0
    skipped_cases = 0
    evaluated_cases = 0

    ground_truth = load_ground_truth_map(GROUND_TRUTH_PATH)

    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)
    cache = load_category_cache(CATEGORY_CACHE_PATH, categories)
    herb_to_category.update(cache)

    if not categories:
        raise RuntimeError("未发现中药大类目录（herb_category/），无法计算 Acc-CL@10")

    fallback_category = (
        DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]
    )

    all_gt_herbs = [h for herbs in ground_truth.values() for h in herbs]
    print("[进度] 预分类 GT 中药...")
    ensure_herbs_categorized(
        all_gt_herbs, herb_to_category, categories,
        CATEGORY_CACHE_PATH, fallback_category,
    )
    print("[进度] GT 中药分类完成")

    rag = SimpleRAG()
    rag.build_index(train_path)

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_cases is not None and idx >= max_cases:
                break

            total_cases += 1
            row = json.loads(line)
            structured = row.get("structured", {})
            case_id = str(row.get("id", "")).strip()

            truth_herbs = ground_truth.get(case_id, [])
            if not truth_herbs:
                if DEBUG:
                    print(f"[DEBUG] 样本 {idx+1}（id={case_id}）无有效真实中药，跳过")
                skipped_cases += 1
                continue

            evaluated_cases += 1
            print(f"\n{'='*60}")
            print(f"[样本 {idx+1}] id={case_id}  已评估={evaluated_cases}")

            similar_cases = rag.retrieve(structured, topk=RAG_TOPK)
            rag_context   = build_rag_context(similar_cases)

            raw_output = run_mdagents(structured, rag_context)

            pred_herbs = extract_herbs_from_output(raw_output, TOPK_HERB)

            # P@10
            p10 = p_at_k(pred_herbs, set(truth_herbs), TOPK_HERB)
            p10_scores.append(p10)

            # Acc-CL@10
            pred_cats  = herbs_to_categories(
                pred_herbs,  herb_to_category, categories,
                CATEGORY_CACHE_PATH, fallback_category,
            )
            truth_cats = herbs_to_categories(
                truth_herbs, herb_to_category, categories,
                CATEGORY_CACHE_PATH, fallback_category,
            )
            p_cl, r_cl, f1_cl = f1_cl_score(pred_cats, truth_cats)
            acc_cl_scores.append(f1_cl)

            print("\n【主任医师最终输出】")
            print(raw_output)
            print(f"\n预测 Top-10：{pred_herbs}")
            print(f"真实中药：  {truth_herbs}")
            print(f"P@10 = {p10:.3f}")
            print(f"预测大类：{pred_cats}")
            print(f"真实大类：{sorted(set(truth_cats))}")
            print(f"Acc-CL@10 = {f1_cl:.3f}  (P={p_cl:.3f}, R={r_cl:.3f})")

    print(f"\n{'='*60}")
    print("====== MDAgents + RAG  最终汇总结果 ======")
    print(f"平均 P@10      = {float(np.mean(p10_scores)):.4f}"
          if p10_scores else "平均 P@10      = 0.0000")
    print(f"平均 Acc-CL@10 = {float(np.mean(acc_cl_scores)):.4f}"
          if acc_cl_scores else "平均 Acc-CL@10 = 0.0000")
    print(f"\n有效样本 = {evaluated_cases} / {total_cases}（跳过 {skipped_cases}）")


# ======================================
# ======================================
if __name__ == "__main__":
    run_experiment(
        test_path=TEST_PATH,
        train_path=TRAIN_PATH,
        max_cases=MAX_TEST_CASE,
    )
