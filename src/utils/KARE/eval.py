"""
prediction/eval.py
════════════════════════════════════════════════════════════════
KARE Step 3c: 评估
────────────────────────────────────────────────────────────────
【直接在 PyCharm 中运行此文件即可】

前置条件：results/predictions.jsonl（generate.py 输出）

输出：results/eval_results.json

评估指标：
  - P@10:      预测top10中命中GT的精确率
  - R@10:      预测top10中命中GT的召回率
  - F1@10:     P@10和R@10的调和均值
  - Acc-CL@10: 基于中药大类的F1（需要 herb_category/ 目录下的 .txt 文件）
════════════════════════════════════════════════════════════════
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent.parent   # prediction/ -> kare_tcm/
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import PREDICT_OUTPUT_FILE, EVAL_OUTPUT_FILE, ROOT_DIR, TOP_K_OUT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERB_CAT_DIR = ROOT_DIR / "herb_category"


def load_herb_categories() -> dict:
    """
    返回 {herb: category} 映射。
    herb_category/ 目录下每个 .txt 文件：
      文件名（不含扩展名）= 大类名，文件内每行一个中药名。
    """
    cat_map = {}
    if HERB_CAT_DIR.exists():
        for fpath in sorted(HERB_CAT_DIR.glob("*.txt")):
            cat = fpath.stem
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    h = line.strip()
                    if h:
                        cat_map[h] = cat
        logger.info(f"Loaded herb categories: {len(cat_map)} herbs, "
                    f"{len(set(cat_map.values()))} categories")
    else:
        logger.warning(f"herb_category/ not found at {HERB_CAT_DIR}, "
                       f"Acc-CL@10 will be skipped")
    return cat_map


def precision_at_k(pred: list, gt: list, k: int = 10) -> float:
    if not gt or not pred:
        return 0.0
    hits = sum(1 for h in pred[:k] if h in set(gt))
    return hits / k


def recall_at_k(pred: list, gt: list, k: int = 10) -> float:
    if not gt or not pred:
        return 0.0
    hits = sum(1 for h in pred[:k] if h in set(gt))
    return hits / len(gt)


def f1_at_k(pred: list, gt: list, k: int = 10) -> float:
    p = precision_at_k(pred, gt, k)
    r = recall_at_k(pred, gt, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def acc_cl_at_k(pred: list, gt: list, herb_cat: dict, k: int = 10) -> float:
    """基于中药大类的 F1"""
    if not herb_cat:
        return float("nan")
    pred_cats = set(herb_cat.get(h, f"__unknown_{h}") for h in pred[:k])
    gt_cats   = set(herb_cat.get(h, f"__unknown_{h}") for h in gt)
    if not pred_cats and not gt_cats:
        return 1.0
    if not pred_cats or not gt_cats:
        return 0.0
    tp   = len(pred_cats & gt_cats)
    prec = tp / len(pred_cats)
    rec  = tp / len(gt_cats)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def main():
    results = []
    with open(PREDICT_OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} prediction results")

    herb_cat = load_herb_categories()

    metrics    = defaultdict(list)
    per_sample = []

    for r in results:
        if r.get("error") or not r.get("pred_herbs"):
            logger.warning(f"  Skip {r['id']}: error or empty prediction")
            continue

        pred = r["pred_herbs"]
        gt   = r["gt_herbs"]
        k    = TOP_K_OUT

        p   = precision_at_k(pred, gt, k)
        rec = recall_at_k(pred, gt, k)
        f1  = f1_at_k(pred, gt, k)
        acc = acc_cl_at_k(pred, gt, herb_cat, k)

        metrics["P@10"].append(p)
        metrics["R@10"].append(rec)
        metrics["F1@10"].append(f1)
        if not np.isnan(acc):
            metrics["Acc-CL@10"].append(acc)

        per_sample.append({
            "id":        r["id"],
            "P@10":      round(p,   4),
            "R@10":      round(rec, 4),
            "F1@10":     round(f1,  4),
            "Acc-CL@10": round(acc, 4) if not np.isnan(acc) else None,
            "pred":      pred,
            "gt":        gt,
        })

    n = len(metrics["P@10"])
    summary = {
        "method":        "KARE-TCM (KG Community Retrieval + Two-Phase Reasoning)",
        "n_evaluated":   n,
        "n_total":       len(results),
        "avg_P@10":      float(np.mean(metrics["P@10"]))       if metrics["P@10"]       else 0,
        "avg_R@10":      float(np.mean(metrics["R@10"]))       if metrics["R@10"]       else 0,
        "avg_F1@10":     float(np.mean(metrics["F1@10"]))      if metrics["F1@10"]      else 0,
        "avg_Acc-CL@10": float(np.mean(metrics["Acc-CL@10"])) if metrics["Acc-CL@10"] else None,
    }

    print("\n" + "═" * 55)
    print("  KARE-TCM Evaluation Results")
    print("═" * 55)
    print(f"  Evaluated:    {n} / {len(results)} samples")
    print(f"  P@10:         {summary['avg_P@10']:.4f}")
    print(f"  R@10:         {summary['avg_R@10']:.4f}")
    print(f"  F1@10:        {summary['avg_F1@10']:.4f}")
    if summary["avg_Acc-CL@10"] is not None:
        print(f"  Acc-CL@10:    {summary['avg_Acc-CL@10']:.4f}")
    else:
        print(f"  Acc-CL@10:    N/A (no herb_category/ found)")
    print("═" * 55 + "\n")

    eval_output = {"summary": summary, "per_sample": per_sample}
    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved evaluation results to {EVAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
