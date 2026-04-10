# KARE-TCM：中医中药推荐系统

## 项目结构

```
kare_tcm/
├── config.py                              ← 全局配置（API Key、路径、超参数）
├── requirements.txt
├── data/                                  ← 数据目录（运行前把数据放这里）
│   └── patients.jsonl                     ← ★ 你的数据文件放这里
├── results/                               ← 自动生成的预测和评估结果
├── logs/                                  ← 日志
├── apis/
│   └── openai_api.py                      ← API 调用封装
├── kg_construction/
│   └── build_kg.py                        ← Step 1：构建知识图谱 + 社区摘要
├── patient_context/
│   ├── base_context.py                    ← Step 2a：构建基础上下文
│   ├── get_emb.py                         ← Step 2b：计算 Embedding
│   ├── sim_patient_ret.py                 ← Step 2c：相似患者检索
│   └── augment_context.py                 ← Step 2d：上下文增强
├── prediction/
│   ├── data_prepare.py                    ← Step 3a：准备推理数据
│   ├── llm_inference/
│   │   └── generate.py                    ← Step 3b：两阶段 LLM 推理
│   └── eval.py                            ← Step 3c：评估
└── herb_category/                         ← （可选）中药大类目录，用于 Acc-CL@10
    └── 补血药.txt                          ← 每个文件：大类名.txt，内容每行一个中药
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

把你的数据文件放到 `data/patients.jsonl`，每行一个 JSON，格式如下：

```json
{"id": "001", "metadata": {"现病史": "...", "四诊(规范)": "...", "病性(泛化)": "心阴虚,肝郁", "病位(泛化)": "心,肝", "中医辨证": "...", "中医诊断": "...", "西医诊断": "...", "中药名称": "麦冬,酸枣仁,..."}}
```

### 3. 在 PyCharm 中按顺序逐步运行（每个文件独立 Run）

| 步骤 | 文件 | 说明 |
|------|------|------|
| Step 1 | `kg_construction/build_kg.py` | 构建知识图谱和 LLM 社区摘要 |
| Step 2a | `patient_context/base_context.py` | 构建每个患者的基础上下文 |
| Step 2b | `patient_context/get_emb.py` | 用 m3e-base 计算 Embedding |
| Step 2c | `patient_context/sim_patient_ret.py` | FAISS / sklearn 相似患者检索 |
| Step 2d | `patient_context/augment_context.py` | 注入 KG 社区摘要，增强上下文 |
| Step 3a | `prediction/data_prepare.py` | 过滤数据，准备推理输入 |
| Step 3b | `prediction/llm_inference/generate.py` | 两阶段 LLM 推理（推理链 + 预测）|
| Step 3c | `prediction/eval.py` | 计算 P@10、R@10、F1@10、Acc-CL@10 |

### 4. 常用配置调整

**只跑前 N 条（调试）**：打开 `generate.py`，在文件末尾修改：
```python
START_IDX = 0
END_IDX   = 10   # 只跑前10条
```

**跳过 LLM 社区摘要（快速验证数据）**：打开 `build_kg.py`，在文件末尾修改：
```python
USE_LLM = False
```

**断点续跑**：设置 `START_IDX = 50`（从第50条开始追加）。

## API 配置

API Key 和 URL 在 `config.py` 中直接配置，也可通过环境变量覆盖：

```bash
export OPENAI_API_KEY=your_key
export OPENAI_API_URL=https://your-api-url/v1/chat/completions
```

Step 1  → kg_construction/build_kg.py
Step 2a → patient_context/base_context.py
Step 2b → patient_context/get_emb.py
Step 2c → patient_context/sim_patient_ret.py
Step 2d → patient_context/augment_context.py
Step 3a → prediction/data_prepare.py
Step 3b → prediction/llm_inference/generate.py
Step 3c → prediction/eval.py
