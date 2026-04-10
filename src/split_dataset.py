import json
import random

# ==========================
# ==========================
INPUT_FILE = "cases_jia_structured.jsonl"
TRAIN_FILE = "train_jia_structured.jsonl"
TEST_FILE = "test_jia_structured.jsonl"

TRAIN_RATIO = 0.8
RANDOM_SEED = 100

# ==========================
# ==========================
def main():
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"📦 总样本数：{len(data)}")

    random.seed(RANDOM_SEED)
    random.shuffle(data)

    split_idx = int(len(data) * TRAIN_RATIO)

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 训练集：{len(train_data)} → {TRAIN_FILE}")
    print(f"✅ 测试集：{len(test_data)} → {TEST_FILE}")
    print("🎯 划分完成（8:2）")

# ==========================
# ==========================
if __name__ == "__main__":
    main()
