# ======================================
# ======================================

import json
import re


INPUT_PATH = "cases_jia.jsonl"
OUTPUT_TXT = "unique_herbs.txt"
OUTPUT_JSON = "unique_herbs.json"


def split_herbs(text: str):
    """
    将 '柴胡, 黄芩, 半夏' / '柴胡，黄芩、半夏' 等
    拆分成单味中药
    """
    if not text:
        return []

    parts = re.split(r"[，,、\s]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_unique_herbs(input_path: str):
    herb_set = set()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            meta = row.get("metadata", {})
            herb_text = meta.get("中药名称", "")


            herbs = split_herbs(herb_text)
            for h in herbs:
                herb_set.add(h)

    return sorted(herb_set)


if __name__ == "__main__":
    herbs = extract_unique_herbs(INPUT_PATH)

    print(f"✅ 共提取到 {len(herbs)} 味不同中药")
    print("示例前 30 味：")
    print(herbs[:30])

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for h in herbs:
            f.write(h + "\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(herbs, f, ensure_ascii=False, indent=2)

    print(f"\n📄 已保存：")
    print(f"- {OUTPUT_TXT}")
    print(f"- {OUTPUT_JSON}")
