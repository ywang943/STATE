# ======================================
# ======================================
import json
import time
import copy
from typing import Dict

import numpy as np
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score


# ======================================
# ======================================
client = OpenAI(
    api_key='xxxYOUR_KEY'
)
MODEL_NAME = "gpt-4o"


# ======================================
# ======================================
VOCAB_PATH = "bingxing_bingwei_vocab_25.json"

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    VOCAB = json.load(f)

BINGWEI_VOCAB = VOCAB.get("病位", [])
BINGXING_VOCAB = VOCAB.get("病性", [])


# ======================================
# ======================================
def extract_json(text: str):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        return json.loads(text[l:r + 1])

    raise ValueError("No valid JSON found")


def llm_json_call(messages, retry=3, max_tokens=300):
    for _ in range(retry):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        raw = resp.choices[0].message.content
        try:
            return raw, extract_json(raw)
        except Exception:
            messages.append({
                "role": "user",
                "content": "上一次输出不是严格 JSON，请只输出 JSON，不要任何解释。"
            })
            time.sleep(1)
    raise RuntimeError("LLM JSON parse failed")


# ======================================
# ======================================
def char_tokenize(text: str):
    return list(text.replace(" ", ""))


def meteor_score(reference: str, hypothesis: str) -> float:
    return sentence_bleu(
        [char_tokenize(reference)],
        char_tokenize(hypothesis),
        smoothing_function=SmoothingFunction().method1
    )


# ======================================
# ======================================
def bertscore(pred: str, truth: str) -> float:
    if not pred or not truth:
        return 0.0
    P, R, F1 = bert_score([pred], [truth], lang="zh", verbose=False)
    return F1.mean().item()


def f1_score(pred: str, truth: str) -> float:
    p = {x.strip() for x in pred.replace("，", ",").split(",") if x.strip()}
    t = {x.strip() for x in truth.replace("，", ",").split(",") if x.strip()}

    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0

    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ======================================
# Prompt
# ======================================
SYSTEM_PROMPT = "你是一名经验丰富的中医辨证论治医生。"

USER_PROMPT_TEMPLATE = """
下面是当前病例的中医【判断线索】（JSON）：
{features}

请你推断该病例的【病位】和【病性】。输出的病位病性可以参考类似于下面的内容但不局限于这些：

病位：{bingwei_vocab}
病性：{bingxing_vocab}

输出要求：
1. 只输出 JSON，不要任何解释
2. JSON 结构必须严格如下：
{{
  "病位": "",
  "病性": ""
}}
"""


# ======================================
# ======================================
def run_experiment(test_path: str, max_cases: int = 20):
    metrics = {
        "bw": {"f1": [], "bert": [], "meteor": []},
        "bx": {"f1": [], "bert": [], "meteor": []}
    }

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_cases:
                break

            s_full = json.loads(line)["structured"]

            true_bw = s_full.get("病位", "")
            true_bx = s_full.get("病性", "")

            if true_bw == "无" or true_bx == "无":
                continue

            s_input = copy.deepcopy(s_full)
            s_input.pop("病位", None)
            s_input.pop("病性", None)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        features=json.dumps(s_input, ensure_ascii=False, indent=2),
                        bingwei_vocab="、".join(BINGWEI_VOCAB),
                        bingxing_vocab="、".join(BINGXING_VOCAB)
                    )
                }
            ]

            raw, pred = llm_json_call(messages)
            pred_bw, pred_bx = pred.get("病位", ""), pred.get("病性", "")

            metrics["bw"]["f1"].append(f1_score(pred_bw, true_bw))
            metrics["bw"]["bert"].append(bertscore(pred_bw, true_bw))
            metrics["bw"]["meteor"].append(meteor_score(true_bw, pred_bw))

            metrics["bx"]["f1"].append(f1_score(pred_bx, true_bx))
            metrics["bx"]["bert"].append(bertscore(pred_bx, true_bx))
            metrics["bx"]["meteor"].append(meteor_score(true_bx, pred_bx))

            print("=" * 80)
            print("GT:", true_bw, true_bx)
            print("PR:", pred_bw, pred_bx)

    print("\n====== No-RAG Overall Evaluation ======")
    for k in ["bw", "bx"]:
        print(f"\n[{ '病位' if k == 'bw' else '病性' }]")
        for m in metrics[k]:
            print(f"{m.upper():8}: {np.mean(metrics[k][m]):.4f}")


# ======================================
# ======================================
if __name__ == "__main__":
    run_experiment(
        test_path="test_jia_structured_filtered_25.jsonl",
        max_cases=20
    )
