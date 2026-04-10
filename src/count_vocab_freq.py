import json
from collections import Counter

# ======================================
# ======================================
INPUT_PATH = "cases_jia_structured.jsonl"
OUTPUT_PATH = "cases_jia_structured_filtered_50.jsonl"
VOCAB_OUTPUT_PATH = "bingxing_bingwei_vocab_50.json"

FIELDS = ["病性", "病位"]

IGNORE_VALUES = {"无", "", None}

FREQ_THRESHOLD = 50


# ======================================
# ======================================
counters = {field: Counter() for field in FIELDS}

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[警告] 第 {line_num} 行 JSON 解析失败：{e}")
            continue

        structured = obj.get("structured", {})

        for field in FIELDS:
            value = structured.get(field)
            if value in IGNORE_VALUES:
                continue

            value = value.replace("，", ",").replace("、", ",")
            terms = [v.strip() for v in value.split(",") if v.strip()]
            counters[field].update(terms)


# ======================================
# ======================================
KEEP_TERMS = {}

for field in FIELDS:
    print(f"\n====== {field} 词频统计 ======")
    print(f"{field}\t出现次数")
    print("-" * 30)

    for term, freq in counters[field].most_common():
        print(f"{term}\t{freq}")

    keep = {
        term for term, freq in counters[field].items()
        if freq > FREQ_THRESHOLD
    }
    KEEP_TERMS[field] = keep

    print(f"\n====== 自动保留的 {field}（freq > {FREQ_THRESHOLD}） ======")
    print("、".join(sorted(keep)) if keep else "（无）")


# ======================================
# ======================================
with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[警告] 第 {line_num} 行 JSON 解析失败：{e}")
            continue

        structured = obj.get("structured", {})

        for field in FIELDS:
            value = structured.get(field)

            if value and value not in IGNORE_VALUES:
                value = value.replace("，", ",").replace("、", ",")
                terms = [v.strip() for v in value.split(",") if v.strip()]
                filtered_terms = [t for t in terms if t in KEEP_TERMS[field]]

                if filtered_terms:
                    structured[field] = ",".join(filtered_terms)
                else:
                    structured[field] = "无"

        obj["structured"] = structured
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ======================================
# ======================================
with open(VOCAB_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {field: sorted(list(terms)) for field, terms in KEEP_TERMS.items()},
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n====== 病性 / 病位 vocab 已保存 ✅ ======")
print(f"输出文件：{VOCAB_OUTPUT_PATH}")

print("\n====== 数据过滤完成 ✅ ======")
print(f"输出文件：{OUTPUT_PATH}")
