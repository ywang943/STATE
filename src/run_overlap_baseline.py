# ======================================
# ======================================
import os
import json
import re
import time
import numpy as np
import requests

# ======================================
# ======================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

TOPK = 10
MAX_TEST_CASE = 8
N_REWRITE = 3

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
def extract_herbs_from_output(text, topk=10):
    herbs = []
    for line in text.splitlines():
        if "中药：" in line:
            line = line.split("中药：", 1)[1]
            parts = re.split(r"[，,、\s]+", line)
            herbs = [p.strip() for p in parts if p.strip()]
            break
    if len(herbs) >= topk:
        herbs = herbs[:topk]
    else:
        herbs += [""] * (topk - len(herbs))
    return herbs


def parse_truth_herbs(text):
    herbs = re.split(r"[，,、\s]+", text)
    return [h.strip() for h in herbs if h.strip()]


def p_at_k(pred, truth_set, k=10):
    hit = sum(1 for h in pred[:k] if h in truth_set)
    return hit / k


def overlap_at_k(pred1, pred2, k=10):
    set1 = set(pred1[:k])
    set2 = set(pred2[:k])
    return len(set1 & set2) / k


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
    "【现病史】\n{case_text}"
)


def rewrite_case_text(case_text, n_rewrites=3):
    rewrites = []
    for _ in range(n_rewrites):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REWRITE_USER_PROMPT.format(case_text=case_text)
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        response = post_with_retry(payload, timeout=500)
        result = response.json()
        rewritten = result["choices"][0]["message"]["content"].strip()
        rewrites.append(rewritten)
    return rewrites


# ======================================
# ======================================
SYSTEM_PROMPT = "你是一名精通中医辨证论治的医生助手。"

USER_QUERY = (
    "请根据以下病历信息进行中医辨证分析，并给出推荐方药。\n"
    "要求：\n"
    "1. 可简要说明辨证思路\n"
    "2. 最后一行必须给出【10味】中药\n"
    "3. 药物格式必须为：\n"
    "中药：药1,药2,药3,药4,药5,药6,药7,药8,药9,药10\n"
)


def recommend_herbs(case_text):
    prompt = f"【病历信息】\n{case_text}\n\n{USER_QUERY}"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    response = post_with_retry(payload, timeout=500)
    result = response.json()
    raw_output = result["choices"][0]["message"]["content"].strip()
    herbs = extract_herbs_from_output(raw_output, TOPK)
    return herbs, raw_output


# ======================================
# ======================================
test_cases = []
with open("test_jia_small.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        test_cases.append(json.loads(line))

print(f"共读取测试病例：{len(test_cases)}")


# ======================================
# ======================================
for idx, case_data in enumerate(test_cases[:MAX_TEST_CASE], 1):
    metadata = case_data.get("metadata", {})
    original_text = metadata.get("现病史", "")
    truth_text = metadata.get("中药名称", "")
    truth_herbs = parse_truth_herbs(truth_text)
    truth_set = set(truth_herbs)

    print("\n" + "=" * 70)
    print(f"样本 {idx}")
    print("原始现病史：\n", original_text)

    orig_pred, orig_output = recommend_herbs(original_text)
    orig_p10 = p_at_k(orig_pred, truth_set)

    print("\n【原始推荐】")
    print(orig_output)
    print("预测中药：", orig_pred)
    print("P@10 =", f"{orig_p10:.3f}")

    rewritten_texts = rewrite_case_text(original_text, N_REWRITE)

    for i, rewrite_text in enumerate(rewritten_texts, 1):
        rewrite_pred, rewrite_output = recommend_herbs(rewrite_text)
        rewrite_p10 = p_at_k(rewrite_pred, truth_set)
        overlap = overlap_at_k(orig_pred, rewrite_pred)

        print("\n" + "-" * 50)
        print(f"【改写版本 {i}】")
        print("改写现病史：\n", rewrite_text)
        print(rewrite_output)
        print("预测中药：", rewrite_pred)
        print(f"P@10 = {rewrite_p10:.3f}")
        print(f"Overlap@10（与原始）= {overlap:.3f}")


# ======================================
# ======================================
all_overlaps = []

for idx, case_data in enumerate(test_cases[:MAX_TEST_CASE], 1):
    metadata = case_data.get("metadata", {})
    original_text = metadata.get("现病史", "")
    truth_text = metadata.get("中药名称", "")
    truth_herbs = parse_truth_herbs(truth_text)
    if not truth_herbs:
        continue

    orig_pred, _ = recommend_herbs(original_text)

    rewritten_texts = rewrite_case_text(original_text, N_REWRITE)
    for rewrite_text in rewritten_texts:
        rewrite_pred, _ = recommend_herbs(rewrite_text)
        overlap = overlap_at_k(orig_pred, rewrite_pred)
        all_overlaps.append(overlap)

if all_overlaps:
    overlaps_np = np.array(all_overlaps, dtype=np.float32)
    print("\n" + "=" * 60)
    print(f"📊 Overlap@10 统计（共 {len(overlaps_np)} 次改写）")
    print(f"标准差 Std  = {overlaps_np.std():.4f}")
    print(f"最小值 Min  = {overlaps_np.min():.4f}")
    print(f"最大值 Max  = {overlaps_np.max():.4f}")
else:
    print("⚠️ 无有效样本")
