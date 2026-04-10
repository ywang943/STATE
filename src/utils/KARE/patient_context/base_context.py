"""
patient_context/base_context.py
════════════════════════════════════════════════════════════════
KARE Step 2a: 构建每个患者的基础上下文（Base Context）
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件：data/patients.jsonl 已存在

输出：data/base_contexts.jsonl
════════════════════════════════════════════════════════════════
"""
import sys
import json
import logging
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import PATIENTS_FILE, BASE_CONTEXT_FILE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_base_context(record: dict) -> dict:
    """把单条患者记录转换为结构化基础上下文。"""
    pid = record.get("id", "")
    m   = record.get("metadata", {})

    xianbing  = m.get("现病史",     "").strip()
    sizhen    = m.get("四诊(规范)", "").strip()
    bingxing  = m.get("病性(泛化)", "").strip()
    bingwei   = m.get("病位(泛化)", "").strip()
    bianzheng = m.get("中医辨证",   "").strip()
    zy_diag   = m.get("中医诊断",   "").strip()
    xy_diag   = m.get("西医诊断",   "").strip()
    herbs_gt  = m.get("中药名称",   "").strip()

    parts = []
    if xianbing:  parts.append(f"现病史：{xianbing}")
    if sizhen:    parts.append(f"四诊（舌脉等）：{sizhen}")
    if bingxing:  parts.append(f"病性：{bingxing}")
    if bingwei:   parts.append(f"病位：{bingwei}")
    if bianzheng: parts.append(f"中医辨证：{bianzheng}")
    if zy_diag:   parts.append(f"中医诊断：{zy_diag}")
    if xy_diag:   parts.append(f"西医诊断：{xy_diag}")

    context_text = "；".join(parts)

    return {
        "id":           pid,
        "context_text": context_text,
        "xianbing":     xianbing,
        "sizhen":       sizhen,
        "bingxing":     bingxing,
        "bingwei":      bingwei,
        "bianzheng":    bianzheng,
        "zy_diag":      zy_diag,
        "xy_diag":      xy_diag,
        "herbs_gt":     herbs_gt,
        "bingxing_list": [x.strip() for x in bingxing.split(",") if x.strip()],
        "bingwei_list":  [x.strip() for x in bingwei.split(",")  if x.strip()],
        "herbs_gt_list": [x.strip() for x in herbs_gt.split(",") if x.strip()],
    }


def main():
    records = []
    with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} patients")

    contexts = [build_base_context(r) for r in records]

    valid   = [c for c in contexts if c["xianbing"]]
    skipped = len(contexts) - len(valid)
    if skipped:
        logger.warning(f"Skipped {skipped} records with empty 现病史")

    with open(BASE_CONTEXT_FILE, "w", encoding="utf-8") as f:
        for ctx in valid:
            f.write(json.dumps(ctx, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(valid)} base contexts to {BASE_CONTEXT_FILE}")


if __name__ == "__main__":
    main()
