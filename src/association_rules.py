# -*- coding: utf-8 -*-
"""
Step1: 基于 transactions.json 做关联分析（FP-Growth）
并保存 Top-20 频繁药组 & 关联规则到 jsonl
"""

import json
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


TRANSACTIONS_PATH = "transactions.json"
OUTPUT_JSONL = "drug_association_top20.jsonl"

MIN_SUPPORT = 0.15
MIN_GROUP_SIZE = 2
TOP_K = 25
# ==========================


def load_transactions(path):
    with open(path, "r", encoding="utf-8") as f:
        transactions = json.load(f)

    print(f"✔ 已加载处方数: {len(transactions)}")
    return transactions


def run_fpgrowth(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    freq_itemsets = fpgrowth(
        df,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    freq_itemsets["size"] = freq_itemsets["itemsets"].apply(len)

    drug_groups = freq_itemsets[
        freq_itemsets["size"] >= MIN_GROUP_SIZE
    ].sort_values("support", ascending=False)

    return freq_itemsets, drug_groups


def run_rules(freq_itemsets):
    rules = association_rules(
        freq_itemsets,
        metric="confidence",
        min_threshold=0.6
    )

    rules = rules[
        (rules["antecedents"].apply(len) >= 1) &
        (rules["consequents"].apply(len) == 1)
    ].sort_values("lift", ascending=False)

    return rules


def save_topk_to_jsonl(drug_groups, rules, out_path):
    records = []

    exp_id = 1

    for rank, (_, row) in enumerate(drug_groups.head(TOP_K).iterrows(), start=1):
        records.append({
            "id": f"EXP_F_{exp_id:04d}",
            "type": "frequent_itemset",
            "rank": rank,
            "items": sorted(list(row["itemsets"])),
            "size": int(row["size"]),
            "support": round(float(row["support"]), 4),
            "status": "active"
        })
        exp_id += 1

    for rank, (_, r) in enumerate(rules.head(TOP_K).iterrows(), start=1):
        records.append({
            "id": f"EXP_R_{exp_id:04d}",
            "type": "association_rule",
            "rank": rank,
            "antecedents": sorted(list(r["antecedents"])),
            "consequents": sorted(list(r["consequents"])),
            "support": round(float(r["support"]), 4),
            "confidence": round(float(r["confidence"]), 4),
            "lift": round(float(r["lift"]), 4),
            "status": "active"
        })
        exp_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✔ 已保存 Top-{TOP_K} 频繁药组 + 关联规则（含ID） 到 {out_path}")



def main():
    if not os.path.exists(TRANSACTIONS_PATH):
        raise FileNotFoundError(f"找不到 {TRANSACTIONS_PATH}")

    transactions = load_transactions(TRANSACTIONS_PATH)
    freq_itemsets, drug_groups = run_fpgrowth(transactions)
    rules = run_rules(freq_itemsets)

    save_topk_to_jsonl(drug_groups, rules, OUTPUT_JSONL)


if __name__ == "__main__":
    main()
