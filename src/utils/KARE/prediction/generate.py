"""
prediction/llm_inference/generate.py
════════════════════════════════════════════════════════════════
KARE Step 3b: 推理增强预测（两阶段：推理链 + 中药预测）
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件：data/predict_input.jsonl（data_prepare.py 输出）

输出：results/predictions.jsonl

配置项（在文件末尾 __main__ 部分修改）：
  START_IDX  — 从第几个样本开始（支持断点续跑，默认 0）
  END_IDX    — 到第几个样本结束（None = 全部）
════════════════════════════════════════════════════════════════
"""
import sys
import json
import re
import time
import logging
import requests
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent.parent.parent   # llm_inference/ -> prediction/ -> kare_tcm/
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import (PREDICT_INPUT_FILE, PREDICT_OUTPUT_FILE,
                    LLM_TEMP_REASON, LLM_TEMP_PREDICT,
                    LLM_MAX_TOKENS, TOP_K_OUT,
                    API_URL, API_KEY, LLM_MODEL,
                    API_MAX_RETRY, API_RETRY_WAIT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def call_llm(messages: list,
             temperature: float = 0.1,
             max_tokens: int = 1200) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY,
    }
    payload = {
        "model":       LLM_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    last_exc = None
    for attempt in range(API_MAX_RETRY):
        try:
            resp = requests.post(API_URL, headers=headers,
                                 json=payload, timeout=500)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_exc = e
            if attempt < API_MAX_RETRY - 1:
                sleep_s = API_RETRY_WAIT * (attempt + 1)
                logger.warning(f"  LLM call failed (attempt {attempt+1}/{API_MAX_RETRY}), "
                               f"retry in {sleep_s}s: {e}")
                time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after {API_MAX_RETRY} attempts. Last: {last_exc}")


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def extract_herbs_from_output(raw_output: str, topk: int = 10) -> list:
    lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if re.search(r"中药[：:]", line):
            content = re.split(r"中药[：:]", line, maxsplit=1)[-1]
            herbs = [h.strip() for h in re.split(r"[,，、\s]+", content) if h.strip()]
            return herbs[:topk]
    if lines:
        herbs = [h.strip() for h in re.split(r"[,，、\s]+", lines[-1]) if h.strip()]
        if len(herbs) >= 3:
            return herbs[:topk]
    return []


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def build_reasoning_messages(sample: dict) -> list:
    system = (
        "你是一名具有30年经验的中医临床专家，精通辨证论治。"
        "你的任务是对患者的证候信息进行深入的辨证分析，"
        "生成详细的中医推理链，为后续选药提供依据。"
        "【重要】本阶段只输出辨证推理分析，不要输出中药列表。"
    )

    user = f"""
=== 知识图谱社区摘要（结构化中医知识）===
{sample.get('community_text', '暂无')}

=== 相似历史病例参考 ===
{sample.get('sim_patients_text', '暂无')}

=== 知识图谱统计推荐中药（供参考）===
{sample.get('kg_herbs_text', '暂无')}

【当前患者】
现病史：{sample.get('xianbing', '')}
四诊（舌脉）：{sample.get('sizhen', '')}
病性：{sample.get('bingxing', '')}
病位：{sample.get('bingwei', '')}
中医辨证：{sample.get('bianzheng', '')}
中医诊断：{sample.get('zy_diag', '')}
西医诊断：{sample.get('xy_diag', '')}

【请按以下结构进行辨证推理分析】：

1. 【主症辨析】分析现病史中的核心症状，明确虚实寒热性质
2. 【四诊合参】结合舌象、脉象等四诊信息，印证或修正判断
3. 【核心病机】基于以上分析，归纳核心病机（如：肾阳亏虚，水湿不化）
4. 【治则治法】根据病机确定具体治则（如：温补肾阳，利水化湿）
5. 【选药方向】明确应选用哪些功效类别的中药（如：补肾阳类、利水类），
   并结合相似病例和知识图谱摘要分析用药规律
6. 【配伍思路】分析君臣佐使，说明各类中药的角色

注意：本步骤只做分析，不直接给出中药名称列表。
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def build_prediction_messages(sample: dict, reasoning_chain: str) -> list:
    system = (
        "你是一名精通中医辨证论治的处方医生。"
        "你将基于已完成的辨证推理分析，选择最适合的10味中药组成处方。"
        "请严格按照格式要求输出。"
    )

    user = f"""
=== 知识图谱社区摘要 ===
{sample.get('community_text', '暂无')}

=== 相似历史病例参考 ===
{sample.get('sim_patients_text', '暂无')}

=== 知识图谱统计推荐中药 ===
{sample.get('kg_herbs_text', '暂无')}

【当前患者】
现病史：{sample.get('xianbing', '')}
四诊：{sample.get('sizhen', '')}
病性：{sample.get('bingxing', '')}
病位：{sample.get('bingwei', '')}
中医辨证：{sample.get('bianzheng', '')}
中医诊断：{sample.get('zy_diag', '')}
西医诊断：{sample.get('xy_diag', '')}

════════════════════════════════════════════
【辨证推理分析（已完成）】
{reasoning_chain}
════════════════════════════════════════════

【选药任务】
基于以上辨证推理链，综合知识图谱推荐和相似病例用药，
选择最适合该患者的恰好10味中药。

选药原则：
- 与推理链中确定的治则高度吻合
- 体现君臣佐使配伍逻辑
- 参考知识图谱高频推荐和相似病例用药
- 避免药性相反的配伍禁忌

**【必须严格按照以下格式输出最后一行，不可缺少】：**
中药：药1, 药2, 药3, 药4, 药5, 药6, 药7, 药8, 药9, 药10
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────
def run_inference(samples: list, start_idx: int = 0, end_idx: int = None) -> list:
    if end_idx is None:
        end_idx = len(samples)

    results = []

    for i, sample in enumerate(samples[start_idx:end_idx], start=start_idx):
        pid = sample["id"]
        logger.info(f"[{i + 1}/{len(samples)}] Patient {pid}")

        try:
            logger.info("  Phase A: reasoning chain generation...")
            reasoning_chain = call_llm(
                build_reasoning_messages(sample),
                temperature=LLM_TEMP_REASON,
                max_tokens=LLM_MAX_TOKENS,
            )
            logger.info(f"  Reasoning chain length: {len(reasoning_chain)} chars")

            logger.info("  Phase B: herb prediction...")
            raw_output = call_llm(
                build_prediction_messages(sample, reasoning_chain),
                temperature=LLM_TEMP_PREDICT,
                max_tokens=LLM_MAX_TOKENS,
            )

            pred_herbs = extract_herbs_from_output(raw_output, topk=TOP_K_OUT)
            logger.info(f"  Predicted: {pred_herbs}")
            logger.info(f"  GT:        {sample.get('herbs_gt_list', [])[:10]}")

            results.append({
                "id":              pid,
                "gt_herbs":        sample.get("herbs_gt_list", []),
                "pred_herbs":      pred_herbs,
                "reasoning_chain": reasoning_chain,
                "raw_output":      raw_output,
                "kg_herbs":        sample.get("kg_herbs", []),
            })

        except Exception as e:
            logger.error(f"  ERROR on patient {pid}: {e}")
            results.append({
                "id":         pid,
                "gt_herbs":   sample.get("herbs_gt_list", []),
                "pred_herbs": [],
                "error":      str(e),
            })

    return results


def main(start_idx: int = 0, end_idx: int = None):
    samples = []
    with open(PREDICT_INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} prediction samples")

    if end_idx is None:
        end_idx = len(samples)
    logger.info(f"Running inference on samples [{start_idx}, {end_idx})")

    results = run_inference(samples, start_idx=start_idx, end_idx=end_idx)

    mode = "a" if start_idx > 0 else "w"
    with open(PREDICT_OUTPUT_FILE, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(results)} predictions to {PREDICT_OUTPUT_FILE}")


if __name__ == "__main__":
    START_IDX = 0
    END_IDX   = None

    main(start_idx=START_IDX, end_idx=END_IDX)
