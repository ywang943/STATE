"""
evaluate.py
===========
Step 5: 批量评估（留一法 Leave-One-Out）

对数据集中每条病历：
  1. 把该病历作为 query
  2. 从其余所有病历中检索（exclude_id 排除自身）
  3. 调用完整 RAG 流程得到推荐中药
  4. 与 Ground Truth 中药对比，计算 P/R/F1

输出：每条记录的评估结果 + 整体平均指标
"""

import json
import os
import time
from main import run_single, load_kg
from retriever import TCMRetriever


def load_data(data_path: str = "data/patients.jsonl") -> list[dict]:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_single(pred_herbs: list[str], gt_herbs: list[str]) -> dict:
    """计算单条的 Precision / Recall / F1"""
    pred_set = set(pred_herbs)
    gt_set   = set(gt_herbs)
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)   if gt_set   else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "hits":      list(pred_set & gt_set),
        "missed":    list(gt_set - pred_set),
        "extra":     list(pred_set - gt_set),
    }


def run_evaluation(
    data_path: str = "data/patients.jsonl",
    output_path: str = "data/eval_results.jsonl",
    max_samples: int = None,
    sleep_between: float = 1.0,
):
    records = load_data(data_path)
    if max_samples:
        records = records[:max_samples]

    print(f"[evaluate] 共 {len(records)} 条病历，开始留一法评估...")

    retriever = TCMRetriever(data_path=data_path)
    kg = load_kg()

    results = []
    all_metrics = []

    for i, rec in enumerate(records):
        pid = rec["id"]
        meta = rec.get("metadata", {})

        patient_history = meta.get("现病史", "")
        gt_herb_str     = meta.get("中药名称", "")
        gt_herbs        = [x.strip() for x in gt_herb_str.split(",") if x.strip()]
        bingxing        = [x.strip() for x in meta.get("病性(泛化)", "").split(",") if x.strip()]
        bingwei         = [x.strip() for x in meta.get("病位(泛化)", "").split(",") if x.strip()]

        if not patient_history or not gt_herbs:
            print(f"[{i+1}/{len(records)}] ID={pid} 跳过（缺少现病史或中药标注）")
            continue

        print(f"\n[{i+1}/{len(records)}] ID={pid}")
        print(f"  现病史: {patient_history[:50]}...")

        try:
            result = run_single(
                patient_history=patient_history,
                patient_bingxing=bingxing,
                patient_bingwei=bingwei,
                retriever=retriever,
                kg=kg,
                exclude_id=pid,
                verbose=False,
            )

            metrics = evaluate_single(result["recommended_herbs"], gt_herbs)

            print(f"  推荐: {result['recommended_herbs']}")
            print(f"  GT:   {gt_herbs}")
            print(f"  P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

            row = {
                "id": pid,
                "patient_history": patient_history,
                "predicted_herbs": result["recommended_herbs"],
                "ground_truth_herbs": gt_herbs,
                **metrics,
                "llm_response": result["llm_response"],
            }
            results.append(row)
            all_metrics.append(metrics)

        except Exception as e:
            print(f"  [错误] {e}")
            results.append({"id": pid, "error": str(e)})

        if sleep_between > 0:
            time.sleep(sleep_between)

    valid = [m for m in all_metrics]
    if valid:
        avg_p  = sum(m["precision"] for m in valid) / len(valid)
        avg_r  = sum(m["recall"]    for m in valid) / len(valid)
        avg_f1 = sum(m["f1"]        for m in valid) / len(valid)

        print("\n" + "="*60)
        print("【整体评估结果】")
        print(f"  样本数:    {len(valid)}")
        print(f"  平均 Precision: {avg_p:.4f}")
        print(f"  平均 Recall:    {avg_r:.4f}")
        print(f"  平均 F1:        {avg_f1:.4f}")
        print("="*60)

        summary = {
            "total": len(valid),
            "avg_precision": avg_p,
            "avg_recall": avg_r,
            "avg_f1": avg_f1,
        }
    else:
        summary = {"total": 0, "error": "no valid results"}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[evaluate] 结果已保存到: {output_path}")
    print(f"[evaluate] 汇总已保存到: {summary_path}")
    return summary


if __name__ == "__main__":
    # run_evaluation(max_samples=5)

    run_evaluation(max_samples=None, sleep_between=1.0)
