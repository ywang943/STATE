# -*- coding: utf-8 -*-
"""
Step0: 从 cases_bumei_structured.jsonl 抽取“推荐中药” -> transactions
"""

import json
import os
from typing import List, Any


INPUT_PATH = "train_jia_structured.jsonl"
OUTPUT_JSON = "transactions.json"
# ==========================


def split_herbs(raw: Any) -> List[str]:
    """把 '药1,药2,药3' -> ['药1','药2','药3']"""
    if raw is None:
        return []

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        for sep in ["，", "、", ";", "；", "|", " "]:
            s = s.replace(sep, ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]
    elif isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
    else:
        s = str(raw).strip()
        for sep in ["，", "、", ";", "；", "|", " "]:
            s = s.replace(sep, ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]

    seen = set()
    out = []
    for h in parts:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def extract_transactions(jsonl_path: str) -> List[List[str]]:
    transactions: List[List[str]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] line {line_no} JSON 解析失败: {e}")
                continue

            structured = obj.get("structured", {})
            raw_herbs = structured.get("推荐中药") if isinstance(structured, dict) else None

            herbs = split_herbs(raw_herbs)
            if herbs:
                transactions.append(herbs)

    print("========== Step0 抽取完成 ==========")
    print(f"输入文件: {jsonl_path}")
    print(f"处方数: {len(transactions)}")
    print("前5条示例：")
    for i, t in enumerate(transactions[:5], 1):
        print(f"  {i}. {t}")

    return transactions


def save_transactions(transactions: List[List[str]], out_path: str):
    dir_path = os.path.dirname(out_path)

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transactions, f, ensure_ascii=False, indent=2)

    print(f"✔ 已保存 transactions 到: {out_path}")


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    transactions = extract_transactions(INPUT_PATH)

    if OUTPUT_JSON:
        save_transactions(transactions, OUTPUT_JSON)


if __name__ == "__main__":
    main()
