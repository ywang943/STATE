# ======================================
# ======================================

import json
import os
import re
import numpy as np
import requests

# ======================================
# ======================================
GPT_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
GPT_API_KEY = ""
GPT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": GPT_API_KEY
}

TOPK_HERB = 10
N_CANDIDATE = 3
MAX_CASE = 150
BATCH_SIZE = 10

EXPERIENCE_PATH = "drug_association_top20_no_init.jsonl"
TRAIN_CASE_PATH = "train_lung_structured.jsonl"

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


def extract_json_from_text(text: str):
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text
    if "```" in text:
        blocks = text.split("```")
        for b in blocks:
            b = b.strip()
            if b.startswith("{") or b.startswith("["):
                return b
    l, r = text.find("["), text.rfind("]")
    if l != -1 and r != -1 and r > l:
        return text[l:r+1]
    return None


# ======================================
# ======================================
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
1. 你**最多只能使用一条经验**，如果使用，第一行必须写明：
   使用经验：[EXP_xxx]
   若未使用经验，请写：使用经验：无
   **注意：经验仅作为参考，不允许完全照抄经验，要有自己的判断和输出内容**
2. 生成一个合理的中药处方，给出10种中药，数量必须是10；
3. 最后一行必须严格输出：
中药：药1,药2,药3,药4,药5,药6,药7,药8,药9,药10
"""

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(GPT_API_URL, headers=GPT_HEADERS, data=json.dumps(data))
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


# ======================================
# ======================================
EVALUATOR_SYSTEM = "你是一名中医处方评估专家。"

def llm_evaluator(pred_text, truth_herbs):
    pred_herbs = extract_herbs_from_output(pred_text, TOPK_HERB)
    hit = sum(1 for h in pred_herbs if h in truth_herbs)

    #threshold = (len(truth_herbs) + 1) // 2
    #label = "positive" if hit >= threshold else "negative"
    label = "positive" if hit >= 5 else "negative"

    return f"Hit: {hit}\nLabel: {label}"


# ======================================
# ======================================
SUMMARIZER_SYSTEM = "你是一个经验库维护专家。"

def llm_summarizer(batch_logs):
    text = "\n\n".join(batch_logs)
    print("看一下batch_logs:", text)

    prompt = f"""
以下是一个 batch 内多个候选处方的生成与评估结果：
{text}

请你总结经验库更新建议，**必须输出 JSON 数组**。

要求：
1. KEEP：仅给 target
2. MODIFY：必须给出新的药组 content，结构需与原经验库一致
   - frequent_itemset: {{items:[...], note:""}}
   - association_rule: {{antecedents:[...], consequents:[...], note:""}}
3. ADD：必须给出完整的新药组或新关联规则，结构同上
4. 如果某个候选处方的 EVAL 多次为 negative，且使用了某条经验（使用经验：[EXP_xxx]），
   可以输出 DELETE 动作删除该经验；若使用经验为“无”，则无需 DELETE
5. 优先 ADD 以下新经验：
   - 在多个 positive 样本的 ground truth 中反复出现
   - 但当前经验库中未出现或极少出现的中药或中药组合
   - 可构造成 frequent_itemset 或 association_rule
   
输出示例：
[
  {{
    "action":"MODIFY",
    "target":"EXP_F_0012",
    "type":"frequent_itemset",
    "content":{{"items":["太子参","白术","茯苓"],"note":"增强健脾利湿"}}
  }},
  {{
    "action":"DELETE",
    "target":"EXP_F_0012",
  }},
  {{
    "action":"ADD",
    "type":"association_rule",
    "content":{{"antecedents":["太子参","白术"],"consequents":["鸡内金"],"note":"高分候选反复出现"}}
  }}
]
"""

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    response = requests.post(GPT_API_URL, headers=GPT_HEADERS, data=json.dumps(data))
    result = response.json()
    raw = result["choices"][0]["message"]["content"]

    print("看一下返回的:", raw)

    json_text = extract_json_from_text(raw)
    if json_text is None:
        print("⚠️ Summarizer 未返回可解析 JSON，跳过")
        return []

    try:
        return json.loads(json_text)
    except Exception as e:
        print("⚠️ JSON 解析失败，跳过")
        print(json_text)
        print(e)
        return []


# ======================================
# ======================================
def apply_controller(exps, actions):
    for act in actions:
        if act["action"] == "DELETE":
            for e in exps:
                if e["id"] == act["target"]:
                    e["status"] = "deleted"

        elif act["action"] == "MODIFY":
            for e in exps:
                if e["id"] == act["target"]:
                    e.update(act["content"])

        elif act["action"] == "ADD":
            exps.append({
                "id": f"EXP_NEW_{len(exps)}",
                "type": act["type"],
                **act["content"],
                "status": "active"
            })
    return exps


# ======================================
# ======================================
def run_pipeline():
    exps = load_experience_base(EXPERIENCE_PATH)
    batch_logs = []
    p10_scores = []

    RETRIEVE_K = 10

    with open(TRAIN_CASE_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    if MAX_CASE:
        cases = cases[:MAX_CASE]

    print(f"测试病例数：{len(cases)}")

    for idx, case in enumerate(cases, 1):
        structured = case["structured"]
        truth_herbs = parse_truth_herbs(structured.get("推荐中药", ""))
        if not truth_herbs:
            continue

        structured_text = build_structured_input(structured)

        for k in range(N_CANDIDATE):
            if len(exps) <= RETRIEVE_K:
                retrieved = exps
            else:
                idxs = np.random.choice(len(exps), size=RETRIEVE_K, replace=False)
                retrieved = [exps[i] for i in idxs]

            gen = llm_generator(structured_text, retrieved)
            pred_herbs = extract_herbs_from_output(gen, TOPK_HERB)
            p10 = p_at_k(pred_herbs, set(truth_herbs), TOPK_HERB)
            p10_scores.append(p10)

            eval_res = llm_evaluator(gen, truth_herbs)

            batch_logs.append(
                f"CASE {idx} | CANDIDATE {k+1}\n"
                f"GEN:\n{gen}\n"
                f"EVAL:\n{eval_res}"
            )

            print("\n" + "=" * 60)
            print(f"样本 {idx} - 候选 {k+1}")
            print(gen)
            print(f"P@10 = {p10:.3f}")

        if idx % BATCH_SIZE == 0:
            actions = llm_summarizer(batch_logs)
            exps = apply_controller(exps, actions)
            batch_logs = []

    if batch_logs:
        actions = llm_summarizer(batch_logs)
        exps = apply_controller(exps, actions)

    with open(EXPERIENCE_PATH, "w", encoding="utf-8") as f:
        for e in exps:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    if p10_scores:
        print("\n" + "=" * 60)
        print(f"平均 P@10 = {np.mean(p10_scores):.4f}")


# ======================================
# ======================================
if __name__ == "__main__":
    run_pipeline()
