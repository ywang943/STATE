"""
prediction/data_prepare.py
════════════════════════════════════════════════════════════════
KARE Step 3a: 准备推理数据
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件：data/augmented_contexts.jsonl（augment_context.py 输出）

输出：data/predict_input.jsonl（过滤掉无 ground-truth 的样本）
════════════════════════════════════════════════════════════════
"""
import sys
import json
import logging
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent.parent   # prediction/ -> kare_tcm/
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import TEST_AUG_CONTEXT_FILE, PREDICT_INPUT_FILE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    samples = []
    with open(TEST_AUG_CONTEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} augmented contexts")

    valid   = [s for s in samples if s.get("herbs_gt_list")]
    skipped = len(samples) - len(valid)
    if skipped:
        logger.warning(f"Skipped {skipped} samples without herbs_gt")

    with open(PREDICT_INPUT_FILE, "w", encoding="utf-8") as f:
        for s in valid:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(valid)} prediction samples to {PREDICT_INPUT_FILE}")


if __name__ == "__main__":
    main()
