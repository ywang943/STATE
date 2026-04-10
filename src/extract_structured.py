# ======================================
# ======================================
import json
import time
import os
import requests

# ======================================
# ======================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ======================================
# ======================================
GPT_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
GPT_API_KEY = ""

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": GPT_API_KEY
}

MODEL_NAME = "gpt-4"

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


def llm_json_call(messages, retry=3, max_tokens=800):
    for _ in range(retry):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens
        }

        response = requests.post(
            GPT_API_URL,
            headers=HEADERS,
            data=json.dumps(payload),
            timeout=300
        )

        response.raise_for_status()
        result = response.json()

        raw = result["choices"][0]["message"]["content"]

        try:
            return extract_json(raw)
        except Exception:
            messages.append({
                "role": "user",
                "content": "上一次输出不是严格 JSON，请只输出 JSON，不要任何解释。"
            })
            time.sleep(1)

    raise RuntimeError("LLM JSON parse failed")

# ======================================
# Prompt
# ======================================
SYSTEM_PROMPT = """
你是一名经验丰富的中医辨证论治助手。
你的任务是从现病史和中医四诊中抽取【判断线索】，并形成一个【经验总结】。
"""

USER_PROMPT_TEMPLATE = """
请根据下面的【现病史】和【中医四诊】完成如下任务：

【任务说明（非常重要）】

你需要完成三步（在脑中完成，不要输出过程）：

1️⃣ 进入“观察子空间”，从中医角度提取判断线索：
- 寒热倾向
- 虚实状态
- 表里层次
- 可能涉及的脏腑与病机
- 症状在时间、饮食、情志、消化等方面的表现

2️⃣ 基于【现病史 + 中医四诊 + 判断线索】，
用 2～3 句话形成一个【中医经验总结描述】：
- 不引入新事实
- 不给出治疗方案
- 不使用药名
- 偏经验性、概括性描述

3️⃣ 给出你对上述【判断线索 + 总结描述】整体成立的【置信分数】：
- 取值范围：0.0 ～ 1.0，可以0.05为一档
- 0.9 及以上表示线索清晰、判断一致
- 0.6～0.8 表示基本合理但存在模糊
- 0.5 以下表示线索不足、判断不稳

【现病史】
{history}

【中医四诊】
{sizhen}

【重要约束（必须严格遵守）】
1. 只输出 JSON，不要任何解释或推理过程
2. “现病史”“中医四诊”“病位”“病性”“推荐中药”字段内容
   必须与原始输入保持一致，不得改写、补充或推断
3. 所有 value 必须是字符串
4. JSON 字段名不可增减
5. 无法判断的字段填写“暂不确定”，但尽量减少该情况

【枚举值约束】
- “寒热”：寒 / 偏寒 / 中性 / 偏热 / 热
- “表里”：表 / 偏表 / 里 / 偏里
- “虚实”：虚 / 偏虚 / 实 / 偏实

【JSON 结构（必须严格一致）】
{{
  "现病史": "",
  "中医四诊": "",

  "寒热": "",
  "虚实": "",
  "表里": "",
  "涉及可能脏腑": "",
  "涉及典型病机": "",
  "动态特征": "",
  "时间节律": "",
  "饮食相关": "",
  "情志相关": "",
  "消化表现": "",

  "总结描述": "",
  "置信分数": "",

  "病位": "",
  "病性": "",
  "推荐中药": ""
}}
"""

# ======================================
# ======================================
def build_structured_case(row: dict) -> dict:
    meta = row.get("metadata", {})

    history = meta.get("现病史", "")
    sizhen = meta.get("中医四诊", "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            history=history,
            sizhen=sizhen
        )}
    ]

    structured = llm_json_call(messages)

    structured["病位"] = meta.get("病位(泛化)", "")
    structured["病性"] = meta.get("病性(泛化)", "")
    structured["推荐中药"] = meta.get("中药名称", "")

    print(structured)
    return structured

# ======================================
# jsonl → structured jsonl
# ======================================
def convert_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)
            structured = build_structured_case(row)

            fout.write(json.dumps({
                "id": row.get("id"),
                "structured": structured
            }, ensure_ascii=False) + "\n")

# ======================================
# ======================================
if __name__ == "__main__":
    convert_jsonl(
        input_path="cases_jia.jsonl",
        output_path="cases_jia_structured.jsonl"
    )
