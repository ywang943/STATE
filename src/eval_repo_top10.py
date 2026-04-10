# ======================================
# ======================================

import json
import re
import numpy as np
import requests

# ======================================
# ======================================
GPT_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
GPT_API_KEY = "874e40a79e924dd8a3695e20b619aaf7121b91563f3b4948af29b5fd10cdffc0"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": GPT_API_KEY
}

MODEL_NAME = "gpt-4"

TOPK_HERB = 10
MAX_CASE = 15

EXPERIENCE_PATH = "drug_association_top20.jsonl"
TEST_CASE_PATH = "test_jia_structured.jsonl"

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
    parts = re.split(r"[，,、\s]+", text)
    return [p.strip() for p in parts if p.strip()]


def p_at_k(pred, truth_set, k=10):
    hit = sum(1 for h in pred[:k] if h in truth_set)
    return hit / k


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

    with open(TEST_CASE_PATH, "r", encoding="utf-8") as f:
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

        hit, label = llm_evaluator(gen, truth_herbs)

        print("\n" + "=" * 60)
        print(f"CASE {idx}")
        print(gen)
        print(f"Hit = {hit}, Label = {label}, P@10 = {p10:.3f}")

    if p10_scores:
        print("\n" + "=" * 60)
        print(f"Test-only 平均 P@10 = {np.mean(p10_scores):.4f}")


# ======================================
# ======================================
if __name__ == "__main__":
    run_test_only()
