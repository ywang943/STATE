"""
main.py (TCM-MedRAG，对齐原始评测脚本)
=======================================

完整流程：
  1. m3e-base 检索 top-K 相似历史病例（RAG）
  2. KG 查询「病性+病位」高频中药
  3. 构造 Prompt（保留原有结构：SYSTEM + USER_QUERY + HERB_CONSTRAINT）
  4. 调用 HKUST GPT API
  5. 评估：P@10 + Acc-CL@10（含大类索引 + LLM分类 + cache）
"""

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
TOP_K_RETRIEVAL = 5
TOP_K_KG        = 12
TOPK_HERB       = 10
MAX_TEST_CASE   = None

# ======================================
# ======================================
GPT_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
GPT_API_KEY = "7bc70a62f11a48c18b00284cac02a7305753c7e4cce748bdb6d80791cfd32459"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": GPT_API_KEY
}
MODEL_NAME = "gpt-4"

# ======================================
# ======================================
SCRIPT_DIR          = Path(__file__).resolve().parent
PROJECT_ROOT        = SCRIPT_DIR

HERB_CATEGORY_DIR   = PROJECT_ROOT / "herb_category"
CATEGORY_CACHE_PATH = PROJECT_ROOT / "herb_category_llm.json"
ENABLE_LLM_CATEGORY = True
CATEGORY_BATCH_SIZE = 20
DEFAULT_CATEGORY    = "无分类"

HERB_STOPWORDS = {"无", "暂无", "无推荐", "无推荐中药"}

# ======================================
# ======================================
DATA_PATH = PROJECT_ROOT / "data" / "test_wei.jsonl"
KG_PATH   = PROJECT_ROOT / "data" / "kg.json"
EMB_PATH  = PROJECT_ROOT / "data" / "embeddings.npz"


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


# ======================================
# ======================================
def normalize_herb_name(name: str) -> str:
    return name.strip()

def parse_truth_herbs(text: str) -> list:
    herbs = re.split(r"[，,、\s]+", text)
    return [h.strip() for h in herbs if h.strip()]

def extract_herbs_from_output(text: str, topk: int = 10) -> list:
    """
    只从「中药：」开头的那一行提取，与原脚本完全一致。
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

def p_at_k(pred: list, truth_set: set, k: int = 10) -> float:
    hit = sum(1 for h in pred[:k] if h in truth_set)
    return hit / k

def f1_cl_score(pred_categories: list, truth_categories: list):
    """基于「大类集合」的 Precision / Recall / F1"""
    P = set(pred_categories)
    G = set(truth_categories)
    if not P or not G:
        return 0.0, 0.0, 0.0
    inter = len(P & G)
    precision = inter / len(P)
    recall    = inter / len(G)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ======================================
# ======================================
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

def classify_herbs_via_llm(herbs: list, categories: List[str], fallback_category: str) -> dict:
    """用 LLM 为缺失大类的中药分类，复用同一套 HKUST API。"""
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
                    {"role": "user",   "content": user},
                ],
                "temperature": 0,
                "max_tokens":  600,
            }
            print("[进度] 开始分类中药批次", flush=True)
            response   = post_with_retry(payload, timeout=500)
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
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

def ensure_herbs_categorized(herbs, herb_to_category, categories, cache_path, fallback_category):
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

def herbs_to_categories(herbs, herb_to_category, categories, cache_path, fallback_category):
    herbs = [normalize_herb_name(h) for h in herbs if normalize_herb_name(h)]
    herb_to_category = ensure_herbs_categorized(
        herbs, herb_to_category, categories, cache_path, fallback_category
    )
    return [herb_to_category.get(herb, fallback_category) for herb in herbs]


# ======================================
# ======================================
_HERB_VOCAB_TEXT = ""
_herb_vocab_path = PROJECT_ROOT / "unique_herbs.txt"
if _herb_vocab_path.exists():
    with open(_herb_vocab_path, "r", encoding="utf-8") as _f:
        _herb_vocab = [line.strip() for line in _f if line.strip()]
    _HERB_VOCAB_TEXT = "，".join(_herb_vocab)

def get_herb_constraint_prompt() -> str:
    if not _HERB_VOCAB_TEXT:
        return ""
    return (
        "【中药使用约束】\n"
        "你只能从下面给定的【中药全集】中选择药物，\n"
        "不得使用列表之外的任何药名（包括别名、炮制名、合成名）。\n\n"
        f"中药全集：\n{_HERB_VOCAB_TEXT}\n"
    )


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
def _load_records(data_path: Path) -> dict:
    records = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[str(rec["id"])] = rec
    return records

def _load_embeddings(emb_path: Path):
    data = np.load(emb_path, allow_pickle=True)
    return data["embeddings"], data["ids"].tolist()

def _load_kg(kg_path: Path) -> dict:
    with open(kg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _retrieve(query_text, model, embeddings, ids, records,
              top_k=5, exclude_id=None) -> list:
    query_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    sims = (embeddings @ query_emb.T).flatten()
    sorted_indices = np.argsort(sims)[::-1]
    results = []
    for idx in sorted_indices:
        pid = ids[idx]
        if pid == exclude_id:
            continue
        if pid not in records:
            continue
        meta = records[pid].get("metadata", {})
        results.append({
            "id":       pid,
            "sim":      float(sims[idx]),
            "现病史":   meta.get("现病史", ""),
            "中药名称": meta.get("中药名称", ""),
            "病性":     meta.get("病性(泛化)", ""),
            "病位":     meta.get("病位(泛化)", ""),
            "中医辨证": meta.get("中医辨证", ""),
        })
        if len(results) >= top_k:
            break
    return results

def _query_kg(kg, bingxing, bingwei, top_k=12) -> list:
    from collections import defaultdict
    herb_score = defaultdict(float)
    for bx in bingxing:
        for bw in bingwei:
            key = f"{bx}|{bw}"
            for herb, cnt in kg.get("combo_herb", {}).get(key, {}).items():
                herb_score[herb] += cnt * 3
    for bx in bingxing:
        for herb, cnt in kg.get("bingxing_herb", {}).get(bx, {}).items():
            herb_score[herb] += cnt * 2
    for bw in bingwei:
        for herb, cnt in kg.get("bingwei_herb", {}).get(bw, {}).items():
            herb_score[herb] += cnt * 1
    return sorted(herb_score.items(), key=lambda x: x[1], reverse=True)[:top_k]

def _format_retrieved(cases: list) -> str:
    lines = []
    for i, c in enumerate(cases, 1):
        lines.append(f"【参考病例 {i}】（相似度: {c['sim']:.3f}）")
        lines.append(f"  现病史: {c['现病史']}")
        if c["病性"]:
            lines.append(f"  病性: {c['病性']}")
        if c["病位"]:
            lines.append(f"  病位: {c['病位']}")
        if c["中医辨证"]:
            lines.append(f"  中医辨证: {c['中医辨证']}")
        lines.append(f"  处方中药: {c['中药名称']}")
        lines.append("")
    return "\n".join(lines)


# ======================================
# ======================================
def build_prompt(structured: dict, retrieved_cases: list, kg_herbs: list) -> str:
    """
    保留原有 USER_QUERY + HERB_CONSTRAINT 结构，
    在【中医判断线索】之前插入 RAG 检索结果和 KG 推荐。
    """
    structured_text = (
        f"现病史: {structured.get('现病史', '')}\n"
        f"中医四诊: {structured.get('中医四诊', '')}\n"
        f"寒热: {structured.get('寒热', '')}\n"
        f"虚实: {structured.get('虚实', '')}\n"
        f"表里: {structured.get('表里', '')}\n"
        f"涉及可能脏腑: {structured.get('涉及可能脏腑', '')}\n"
        f"涉及典型病机: {structured.get('涉及典型病机', '')}\n"
        f"动态特征: {structured.get('动态特征', '')}\n"
        f"时间节律: {structured.get('时间节律', '')}\n"
        f"饮食相关: {structured.get('饮食相关', '')}\n"
        f"情志相关: {structured.get('情志相关', '')}\n"
        f"消化表现: {structured.get('消化表现', '')}\n"
    )

    rag_section = ""
    if retrieved_cases:
        rag_section = (
            "=== 相似历史病例参考（RAG检索）===\n"
            + _format_retrieved(retrieved_cases)
            + "\n"
        )

    kg_section = ""
    if kg_herbs:
        herb_str = "、".join([h for h, _ in kg_herbs[:8]])
        kg_section = (
            "=== 知识图谱统计推荐 ===\n"
            f"基于病性/病位统计，以下中药出现频率较高：{herb_str}\n\n"
        )

    herb_constraint = get_herb_constraint_prompt()

    return (
        f"{rag_section}"
        f"{kg_section}"
        f"【中医判断线索】\n"
        f"{structured_text}\n"
        f"{herb_constraint}"
        f"{USER_QUERY}"
    )


# ======================================
# ======================================
def metadata_to_structured(meta: dict) -> dict:
    """
    把你的 metadata 字段映射到原脚本的 structured 字段。
    没有对应字段的留空，LLM 会自动忽略空行。
    """
    return {
        "现病史":       meta.get("现病史", ""),
        "中医四诊":     meta.get("四诊(规范)", meta.get("中医四诊", "")),
        "推荐中药":     meta.get("中药名称", ""),
        "涉及可能脏腑": meta.get("病位(泛化)", ""),
        "涉及典型病机": meta.get("病性(泛化)", ""),
        "寒热":         meta.get("寒热", ""),
        "虚实":         meta.get("虚实", ""),
        "表里":         meta.get("表里", ""),
        "动态特征":     meta.get("动态特征", ""),
        "时间节律":     meta.get("时间节律", ""),
        "饮食相关":     meta.get("饮食相关", ""),
        "情志相关":     meta.get("情志相关", ""),
        "消化表现":     meta.get("消化表现", ""),
    }


# ======================================
# ======================================
def run_experiment(input_path: str = None, sleep_between: float = 1.0):
    data_file  = Path(input_path) if input_path else DATA_PATH
    test_cases = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))

    if MAX_TEST_CASE:
        test_cases = test_cases[:MAX_TEST_CASE]

    print(f"测试病例数：{len(test_cases)}")

    print("[init] 加载 m3e-base embedding 模型...")
    from sentence_transformers import SentenceTransformer
    emb_model  = SentenceTransformer("moka-ai/m3e-base")
    records    = _load_records(DATA_PATH)
    embeddings, emb_ids = _load_embeddings(EMB_PATH)
    kg         = _load_kg(KG_PATH)

    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)
    if not categories:
        print("[WARN] 未发现 herb_category 目录，Acc-CL@10 将跳过")
        skip_cl = True
        fallback_category = DEFAULT_CATEGORY
    else:
        skip_cl = False
        fallback_category = (
            DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]
        )
        cache = load_category_cache(CATEGORY_CACHE_PATH, categories)
        herb_to_category.update(cache)

        all_gt_herbs = []
        for row in test_cases:
            meta = row.get("metadata", {})
            truth_text = meta.get("中药名称", "")
            all_gt_herbs.extend([
                normalize_herb_name(h)
                for h in parse_truth_herbs(truth_text)
                if normalize_herb_name(h) and normalize_herb_name(h) not in HERB_STOPWORDS
            ])
        if all_gt_herbs:
            print("[进度] 开始加载真实值类别索引", flush=True)
            ensure_herbs_categorized(
                all_gt_herbs, herb_to_category, categories,
                CATEGORY_CACHE_PATH, fallback_category
            )
            print("[进度] 真实值类别索引完成", flush=True)

    p10_scores      = []
    acc_cl_scores   = []
    total_cases     = 0
    skipped_cases   = 0
    evaluated_cases = 0

    for idx, row in enumerate(test_cases, 1):
        total_cases += 1
        meta       = row.get("metadata", {})
        structured = metadata_to_structured(meta)
        pid        = str(row["id"])

        # Ground Truth
        truth_text  = structured.get("推荐中药", "")
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

        patient_history = structured.get("现病史", "")
        retrieved = _retrieve(
            patient_history, emb_model, embeddings, emb_ids,
            records, top_k=TOP_K_RETRIEVAL, exclude_id=pid
        )

        bingxing = [x.strip() for x in meta.get("病性(泛化)", "").split(",") if x.strip()]
        bingwei  = [x.strip() for x in meta.get("病位(泛化)", "").split(",") if x.strip()]
        if not bingxing and retrieved:
            bingxing = [x.strip() for x in retrieved[0].get("病性", "").split(",") if x.strip()]
        if not bingwei and retrieved:
            bingwei  = [x.strip() for x in retrieved[0].get("病位", "").split(",") if x.strip()]
        kg_herbs = _query_kg(kg, bingxing, bingwei, top_k=TOP_K_KG)

        user_content = build_prompt(structured, retrieved, kg_herbs)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens":  1000,
        }
        response   = post_with_retry(payload, timeout=500)
        raw_output = response.json()["choices"][0]["message"]["content"].strip()
        pred_herbs = extract_herbs_from_output(raw_output, TOPK_HERB)

        # P@10
        p10 = p_at_k(pred_herbs, truth_set, TOPK_HERB)
        p10_scores.append(p10)

        # Acc-CL@10
        if not skip_cl:
            pred_categories  = herbs_to_categories(
                pred_herbs,  herb_to_category, categories, CATEGORY_CACHE_PATH, fallback_category
            )
            truth_categories = herbs_to_categories(
                truth_herbs, herb_to_category, categories, CATEGORY_CACHE_PATH, fallback_category
            )
            p_cl, r_cl, f1_cl = f1_cl_score(pred_categories, truth_categories)
            acc_cl_scores.append(f1_cl)
        else:
            pred_categories = truth_categories = []
            p_cl = r_cl = f1_cl = 0.0

        print("\n" + "=" * 60)
        print(f"样本 {idx}")
        print("模型输出：")
        print(raw_output)
        print("\n预测 Top-10：", pred_herbs)
        print("真实中药：",     truth_herbs)
        print(f"P@10 = {p10:.3f}")
        if not skip_cl:
            print("预测大类 Top-10：", pred_categories)
            print("真实大类：",        sorted(set(truth_categories)))
            print(f"CL-Precision = {p_cl:.3f}")
            print(f"CL-Recall    = {r_cl:.3f}")
            print(f"CL-F1        = {f1_cl:.3f}")

        if sleep_between > 0:
            time.sleep(sleep_between)

    print("\n" + "=" * 60)
    if p10_scores:
        print(f"平均 P@10      = {float(np.mean(p10_scores)):.4f}")
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
    run_experiment()