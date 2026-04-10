import pandas as pd
import json
import argparse

def clean_record(row, idx):
    """
    把一行病例记录转成统一的文本格式
    """
    parts = []
    mapping = {
        "现病史": "现病史",
        "现病史简化": "现病史简化",
        "现病史极简": "现病史极简",
        "临床表现(标杆词)": "临床表现(标杆词)",
        "病性(泛化)": "病性(泛化)",
        "病位(泛化)": "病位(泛化)",
        "中医辨证": "中医辨证",
        "中医四诊": "中医四诊",
        "中医四诊(洗)": "中医四诊(洗)",
        "四诊(规范)": "四诊(规范)",
        "中医诊断": "中医诊断",
        "西医诊断": "西医诊断",
        "大模型53病性病位": "大模型53病性病位",
        "大模型53病性病位(>=18)": "大模型53病性病位(>=18)",
        "大模型复合中医病性": "大模型复合中医病性",
        "中药名称": "中药名称"
    }

    metadata = {}
    for k, label in mapping.items():
        val = row[k] if (k in row and pd.notna(row[k])) else "无"
        parts.append(f"{label}: {val}")
        metadata[k] = val

    text = "；".join(parts)
    return {
        "id": str(idx),
        "text": text,
        "metadata": metadata
    }

def main(input_file, output_file):
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    records = []
    for i, row in df.iterrows():
        record = clean_record(row, idx=i+1)
        records.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 已保存 {len(records)} 条病例到 {output_file}")

if __name__ == "__main__":
    INPUT_FILE = "data_jia.xlsx"
    OUTPUT_FILE = "cases_jia.jsonl"

    main(INPUT_FILE, OUTPUT_FILE)
