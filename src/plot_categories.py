import json
import re
from pathlib import Path
from collections import Counter

# =========================
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "cases_jia_structured.jsonl"
HERB_CATEGORY_PATH = PROJECT_ROOT / "herb_to_category_full.json"

# =========================
# =========================
def normalize_herb_name(name: str) -> str:
    return name.strip()

def parse_herbs(text: str):
    return [
        normalize_herb_name(h)
        for h in re.split(r"[，,、\s]+", text)
        if h.strip()
    ]

# =========================
# =========================
def main():
    print("[1] 加载 药 → 大类 映射")
    herb_to_category = json.load(
        open(HERB_CATEGORY_PATH, encoding="utf-8")
    )

    print("[2] 统计所有中药出现次数")
    herb_counter = Counter()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            herb_text = row.get("structured", {}).get("推荐中药", "")
            herbs = parse_herbs(herb_text)
            herb_counter.update(herbs)

    print(f"    共出现中药 {len(herb_counter)} 种")

    print("[3] 映射到大类并统计频次")
    category_counter = Counter()

    missing = set()

    for herb, cnt in herb_counter.items():
        category = herb_to_category.get(herb)
        if category is None:
            missing.add(herb)
            continue
        category_counter[category] += cnt

    # =========================
    # =========================
    print("\n====== 中药出现频次 Top 20 ======")
    for herb, cnt in herb_counter.most_common(20):
        print(f"{herb:10s} {cnt}")

    print("\n====== 大类出现频次 ======")
    for cat, cnt in category_counter.most_common():
        print(f"{cat:15s} {cnt}")

    print("\n====== 未找到大类的中药 ======")
    if missing:
        for h in sorted(missing):
            print(h)
    else:
        print("无")

    out = {
        "herb_frequency": herb_counter,
        "category_frequency": category_counter
    }

    out_path = PROJECT_ROOT / "category_frequency.json"
    out_path.write_text(
        json.dumps(
            {
                "herb_frequency": dict(herb_counter),
                "category_frequency": dict(category_counter),
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"\n✅ 统计结果已保存：{out_path}")

if __name__ == "__main__":
    main()
