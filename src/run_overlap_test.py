# ======================================
# ======================================
import json
import time
import re
import os
import sys
import io
import copy
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
LOG_FILE = LOG_DIR / f"hyper_overlap_{time.strftime('%Y%m%d_%H%M%S')}.log"


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
MAX_TEST_CASE = 15
N_REWRITE = 3

EXPERIENCE_PATH = SCRIPT_DIR / "drug_association_top20.jsonl"
HYPEREDGES_PATH = SCRIPT_DIR / "hyperedges.jsonl"
TEST_PATH = SCRIPT_DIR / "test_jia_structured.jsonl"
TRAIN_PATH = SCRIPT_DIR / "train_jia_structured.jsonl"

GROUND_TRUTH_PATH = PROJECT_ROOT / "cases_jia.jsonl"


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


def overlap_at_k(pred1, pred2, k=10):
    s1 = set([h for h in pred1[:k] if h])
    s2 = set([h for h in pred2[:k] if h])
    return len(s1 & s2) / k


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
    experience_text = "\n".join(json.dumps(e, ensure_ascii=False) for e in experiences)

    prompt = EXPERIENCE_DISTILL_PROMPT.format(
        features=json.dumps(structured_features, ensure_ascii=False, indent=2),
        experience_text=experience_text,
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": EXPERIENCE_DISTILL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 350,
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
REWRITE_SYSTEM_PROMPT = "你是一名医学文本规范化与等价改写专家。"

REWRITE_USER_PROMPT = (
    "请对以下【现病史】进行改写，要求：\n"
    "1. 只改变表述方式，不改变医学含义\n"
    "2. 不新增、不删除任何症状或信息\n"
    "3. 不引入新的中医或西医判断\n"
    "4. 只输出改写后的文本，不要解释\n"
    "5. 在不改变含义的前提下，可以主要适当调整句子或信息的先后顺序\n\n"
    "【现病史】\n{text}"
)


def rewrite_present_illness(text: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": REWRITE_USER_PROMPT.format(text=text)},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    response = post_with_retry(payload, timeout=500)
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def rewrite_structured_only_xbs(structured: dict) -> dict:
    """
    只改写 structured['现病史']，其他字段保持完全不变
    """
    new_struct = copy.deepcopy(structured)
    xbs = structured.get("现病史", "")
    if isinstance(xbs, str) and xbs.strip():
        try:
            new_struct["现病史"] = rewrite_present_illness(xbs.strip())
        except Exception as e:
            print(f"[WARN] 现病史改写失败，回退原文: {e}")
            new_struct["现病史"] = xbs
    return new_struct


# ======================================
# ======================================
def infer_one_case(
    structured: dict,
    edges,
    embed_model,
    field_to_entities,
    field_to_index,
    summary_index,
    experiences,
):
    structured_for_prompt = {
        k: v for k, v in structured.items() if k not in ["病位", "病性", "推荐中药"]
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
        topk_per_field=2,
    )

    rag_context_text = build_rag_context_text(rag_results)

    print("[进度] 开始 LLM 总结经验", flush=True)
    experience_summary = distill_experience_by_llm(
        structured_features=structured_for_prompt, experiences=experiences
    )
    print("[进度] LLM 总结经验完成", flush=True)

    if not experience_summary.strip():
        experience_summary = "无可用经验"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        features=json.dumps(structured_for_prompt, ensure_ascii=False, indent=2),
        rag_context=rag_context_text,
        experience_summary=experience_summary,
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    print("[进度] 开始 LLM 生成推荐", flush=True)
    response = post_with_retry(payload, timeout=500)
    result = response.json()
    raw_output = result["choices"][0]["message"]["content"].strip()
    print("[进度] LLM 生成推荐完成", flush=True)

    pred_herbs = extract_herbs_from_output(raw_output, TOPK_HERB)
    return pred_herbs, raw_output


# ======================================
# ======================================
def run_experiment(test_path, train_path, max_cases=100, n_rewrite=3):
    test_path = resolve_path(test_path, SCRIPT_DIR, PROJECT_ROOT)
    train_path = resolve_path(train_path, SCRIPT_DIR, PROJECT_ROOT)

    ground_truth = load_ground_truth_map(GROUND_TRUTH_PATH)
    experiences = load_experience_base(EXPERIENCE_PATH)

    print("[进度] 开始加载超图", flush=True)
    edges = load_hyperedges(str(HYPEREDGES_PATH))
    (embed_model, field_to_entities, field_to_index, summary_index) = build_indexes(edges)
    print("[进度] 超图索引/向量构建完成", flush=True)

    p10_scores_orig = []
    p10_scores_rewrite = []
    overlaps = []

    total_cases = 0
    skipped_cases = 0
    evaluated_cases = 0

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

            truth_set = set(truth_herbs)
            evaluated_cases += 1

            print("\n" + "=" * 80)
            print(f"[样本 {evaluated_cases}] id={case_id}")
            print("原始现病史：", structured.get("现病史", ""))

            orig_pred, orig_output = infer_one_case(
                structured=structured,
                edges=edges,
                embed_model=embed_model,
                field_to_entities=field_to_entities,
                field_to_index=field_to_index,
                summary_index=summary_index,
                experiences=experiences,
            )
            orig_p10 = p_at_k(orig_pred, truth_set, TOPK_HERB)
            p10_scores_orig.append(orig_p10)

            print("\n【原始输出】")
            print(orig_output)
            print("原始 Top-10：", orig_pred)
            print(f"P@10 = {orig_p10:.3f}")

            for r_i in range(1, n_rewrite + 1):
                print("\n" + "-" * 60)
                print(f"[改写版本 {r_i}] 开始改写现病史", flush=True)

                rewritten_struct = rewrite_structured_only_xbs(structured)
                print("改写现病史：", rewritten_struct.get("现病史", ""))

                rewrite_pred, rewrite_output = infer_one_case(
                    structured=rewritten_struct,
                    edges=edges,
                    embed_model=embed_model,
                    field_to_entities=field_to_entities,
                    field_to_index=field_to_index,
                    summary_index=summary_index,
                    experiences=experiences,
                )

                rewrite_p10 = p_at_k(rewrite_pred, truth_set, TOPK_HERB)
                p10_scores_rewrite.append(rewrite_p10)

                ov = overlap_at_k(orig_pred, rewrite_pred, TOPK_HERB)
                overlaps.append(ov)

                print("\n【改写输出】")
                print(rewrite_output)
                print("改写 Top-10：", rewrite_pred)
                print(f"P@10 = {rewrite_p10:.3f}")
                print(f"Overlap@10（与原始）= {ov:.3f}")

    print("\n" + "=" * 80)
    print("====== 最终统计（只测 P@10 + Overlap@10） ======")
    print(f"有效样本数 = {evaluated_cases} / {total_cases} (跳过 {skipped_cases})")

    if p10_scores_orig:
        print(f"原始 平均 P@10 = {float(np.mean(p10_scores_orig)):.4f}")
    else:
        print("原始 平均 P@10 = 0.0000")

    if p10_scores_rewrite:
        print(f"改写 平均 P@10 = {float(np.mean(p10_scores_rewrite)):.4f}")
    else:
        print("改写 平均 P@10 = 0.0000")

    if overlaps:
        print(f"平均 Overlap@10 = {float(np.mean(overlaps)):.4f}  (共 {len(overlaps)} 次改写)")
        print(f"Overlap@10 Std = {float(np.std(overlaps)):.4f}")
        print(f"Overlap@10 Min/Max = {float(np.min(overlaps)):.4f} / {float(np.max(overlaps)):.4f}")
    else:
        print("平均 Overlap@10 = 0.0000")


# ======================================
# ======================================
if __name__ == "__main__":
    run_experiment(
        test_path=TEST_PATH,
        train_path=TRAIN_PATH,
        max_cases=MAX_TEST_CASE,
        n_rewrite=N_REWRITE,
    )
