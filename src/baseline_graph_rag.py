# ======================================
# ======================================
import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from pathlib import Path
from collections import Counter

import numpy as np
import requests
import networkx as nx
from sentence_transformers import SentenceTransformer

# ======================================
# ======================================
GPT_API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"

GPT_API_KEY = "874e40a79e924dd8a3695e20b619aaf7121b91563f3b4948af29b5fd10cdffc0"

MODEL_NAME = "gpt-4"

TOPK_RAG = 5
TOPK_HERB = 10
MAX_TEST_CASE = 100
DEBUG = True

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "graphrag_cache"
CACHE_DIR.mkdir(exist_ok=True)
COMMUNITY_CACHE_PATH = CACHE_DIR / "community_corpus_cache.json"  # summary + herbs + entities
EMBED_CACHE_PATH = CACHE_DIR / "community_embeds.npy"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": GPT_API_KEY
}

# ======================================
# ======================================
with open("unique_herbs.txt", "r", encoding="utf-8") as f:
    herb_vocab = [line.strip() for line in f if line.strip()]
HERB_VOCAB_TEXT = "，".join(herb_vocab)
HERB_SET = set(herb_vocab)

# ======================================
# ======================================
def extract_herbs_from_output(text: str, topk: int = 10) -> List[str]:
    """从 LLM 输出中提取最后一行 ‘中药：...’ """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("中药："):
            herbs = re.split(r"[，,、\s]+", line[len("中药："):])
            herbs = [h.strip() for h in herbs if h.strip()]
            herbs = herbs[:topk]
            herbs += [""] * (topk - len(herbs))
            return herbs
    return [""] * topk


def parse_truth_herbs(text: str) -> List[str]:
    herbs = [h.strip() for h in re.split(r"[，,、\s]+", text or "") if h.strip()]
    herbs = [h for h in herbs if h in HERB_SET]
    return herbs


def p_at_k(pred: List[str], truth_set: set, k: int = 10) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for h in pred[:k] if h in truth_set) / k


def build_query_text(structured: dict) -> str:
    parts = []
    for k, v in structured.items():
        if isinstance(v, list):
            v = "、".join(v)
        parts.append(f"{k}：{v}")
    return "\n".join(parts)

# ======================================
# ======================================
HERB_CATEGORY_DIR = SCRIPT_DIR / "herb_category"
CATEGORY_CACHE_PATH = SCRIPT_DIR / "herb_category_llm.json"
DEFAULT_CATEGORY = "无分类"
ENABLE_LLM_CATEGORY = True


def normalize_herb_name(name: str) -> str:
    return name.strip()


def load_category_index(root_dir: Path):
    herb_to_categories = {}
    categories = set()
    if not root_dir.exists():
        return {}, []

    for path in root_dir.rglob("*.txt"):
        category = path.parent.name
        herb = normalize_herb_name(path.stem)
        categories.add(category)
        herb_to_categories.setdefault(herb, set()).add(category)

    herb_to_category = {
        herb: sorted(list(cats))[0]
        for herb, cats in herb_to_categories.items()
    }
    return herb_to_category, sorted(categories)


def load_category_cache(path: Path, categories: List[str]):
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    cache = {}
    for herb, cats in raw.items():
        herb = normalize_herb_name(herb)
        if isinstance(cats, list):
            for c in cats:
                if c in categories:
                    cache[herb] = c
                    break
        elif cats in categories:
            cache[herb] = cats
    return cache


def save_category_cache(path: Path, cache: dict):
    payload = {k: [v] for k, v in cache.items()}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def f1_cl_score(pred_categories, truth_categories):
    """
    基于「大类集合」的 Precision / Recall / F1
    """
    P = set(pred_categories)
    G = set(truth_categories)

    if not P or not G:
        return 0.0, 0.0, 0.0

    inter = len(P & G)
    precision = inter / len(P)
    recall = inter / len(G)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def classify_herbs_via_llm(herbs, categories, fallback):
    system = (
        "你是中药分类助手。"
        "请从给定的大类中为每味中药选择最合适的1个类别。"
        "只能输出 JSON。"
    )
    category_text = "、".join(categories)
    results = {}

    for herb in herbs:
        user = (
            f"大类列表：{category_text}\n"
            f"中药：{herb}\n"
            "输出示例：{\"中药名\": \"大类\"}"
        )
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0,
            "max_tokens": 200
        }
        try:
            r = requests.post(GPT_API_URL, headers=HEADERS, data=json.dumps(payload), timeout=300)
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            obj = json.loads(re.search(r"\{.*\}", text, re.S).group())
            cat = list(obj.values())[0]
            results[herb] = cat if cat in categories else fallback
        except Exception:
            results[herb] = fallback

    return results


def ensure_herbs_categorized(herbs, herb_to_category, categories, cache_path, fallback):
    missing = [h for h in herbs if h and h not in herb_to_category]
    if not missing:
        return herb_to_category

    if ENABLE_LLM_CATEGORY:
        new_map = classify_herbs_via_llm(missing, categories, fallback)
    else:
        new_map = {h: fallback for h in missing}

    herb_to_category.update(new_map)

    cache = load_category_cache(cache_path, categories)
    cache.update(new_map)
    save_category_cache(cache_path, cache)

    return herb_to_category


def herbs_to_categories(herbs, herb_to_category, categories, cache_path, fallback):
    herbs = [normalize_herb_name(h) for h in herbs if h]
    herb_to_category = ensure_herbs_categorized(
        herbs, herb_to_category, categories, cache_path, fallback
    )
    return [herb_to_category.get(h, fallback) for h in herbs]

# ======================================
# ======================================
def extract_entities_from_structured(structured: dict) -> List[str]:
    """
    把 structured 里的字段值都当作“实体候选”。
    注意：中药不要在这里混进来（我们用社区属性方式注入中药）
    """
    entities = set()
    for k, v in structured.items():
        if k == "推荐中药":
            continue

        if isinstance(v, list):
            entities.update([str(x).strip() for x in v if str(x).strip()])
        elif isinstance(v, str):
            vv = v.strip()
            if vv:
                entities.add(vv)

    entities = [e for e in entities if e and len(e) <= 30]
    return entities


# ======================================
# ======================================
def build_entity_graph(train_path: str) -> nx.Graph:
    G = nx.Graph()

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            structured = row["structured"]
            entities = extract_entities_from_structured(structured)

            for e in entities:
                G.add_node(e)

            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    u, v = entities[i], entities[j]
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += 1
                    else:
                        G.add_edge(u, v, weight=1)

    return G


# ======================================
# ======================================
def build_communities(G: nx.Graph) -> List[List[str]]:
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    return [list(c) for c in communities]


# ======================================
# ======================================
def call_llm(system: str, user: str, temperature: float = 0.0, max_tokens: int = 300) -> str:
    if not GPT_API_KEY:
        raise RuntimeError("GPT_API_KEY 未设置。请用环境变量 GPT_API_KEY 或在代码里填写。")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    r = requests.post(GPT_API_URL, headers=HEADERS, data=json.dumps(payload), timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def summarize_community(entities: List[str]) -> str:
    system = "你是中医临床知识总结助手。"
    user = (
        "以下是一组在同一批病例中频繁共现的医学信息片段（症状/体征/描述）。\n"
        "请总结它们共同对应的中医证候、病机要点与治疗思路（不要编造具体病例）。\n\n"
        f"片段：{'，'.join(entities[:120])}\n\n"
        "输出要求：用简明条目总结（3-8条）。"
    )
    return call_llm(system, user, temperature=0.0, max_tokens=350)


# ======================================
# ======================================
def collect_community_herbs(
    community_entities: List[str],
    train_path: str,
    topk: int = 20,
    min_overlap: int = 5
) -> List[str]:
    """
    遍历训练集，如果某病例实体与该社区实体有 overlap，
    则把该病例的 推荐中药 计入该社区的 herb counter。
    """
    counter = Counter()
    comm_set = set(community_entities)

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            structured = row["structured"]
            case_entities = extract_entities_from_structured(structured)

            if len(comm_set.intersection(case_entities)) >= min_overlap:
                herbs = parse_truth_herbs(structured.get("推荐中药", ""))
                counter.update([h for h in herbs if h in HERB_SET])

    herbs = [h for h, _ in counter.most_common(topk)]
    return herbs


# ======================================
# ======================================
@dataclass
class CommunityDoc:
    entities: List[str]
    summary: str
    herbs: List[str]


def save_corpus(path: Path, corpus: List[CommunityDoc]) -> None:
    payload = [
        {"entities": c.entities, "summary": c.summary, "herbs": c.herbs}
        for c in corpus
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus(path: Path) -> List[CommunityDoc]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [CommunityDoc(**x) for x in raw]


def build_graph_rag_corpus(train_path: str, force_rebuild: bool = False) -> List[CommunityDoc]:
    if COMMUNITY_CACHE_PATH.exists() and not force_rebuild:
        if DEBUG:
            print(f"[CACHE] 使用已缓存社区语料：{COMMUNITY_CACHE_PATH}")
        return load_corpus(COMMUNITY_CACHE_PATH)

    print("[GraphRAG] 构建实体图...")
    G = build_entity_graph(train_path)

    print("[GraphRAG] 社区发现...")
    communities = build_communities(G)
    print(f"[GraphRAG] 社区数量：{len(communities)}")

    corpus: List[CommunityDoc] = []
    for idx, comm_entities in enumerate(communities, 1):
        comm_entities = comm_entities[:10]

        print(f"[GraphRAG] Summarize community {idx}/{len(communities)} (entities={len(comm_entities)})")
        summary = summarize_community(comm_entities)

        herbs = collect_community_herbs(comm_entities, train_path, topk=30, min_overlap=1)

        corpus.append(CommunityDoc(
            entities=comm_entities,
            summary=summary,
            herbs=herbs
        ))

    save_corpus(COMMUNITY_CACHE_PATH, corpus)
    print(f"[GraphRAG] 已缓存社区语料到：{COMMUNITY_CACHE_PATH}")
    return corpus


# ======================================
# ======================================
def build_or_load_embeddings(
    corpus: List[CommunityDoc],
    embed_model: SentenceTransformer,
    force_rebuild: bool = False
) -> np.ndarray:
    if EMBED_CACHE_PATH.exists() and not force_rebuild:
        if DEBUG:
            print(f"[CACHE] 使用已缓存 embeddings：{EMBED_CACHE_PATH}")
        return np.load(EMBED_CACHE_PATH)

    summaries = [c.summary for c in corpus]
    embeds = embed_model.encode(summaries, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBED_CACHE_PATH, embeds)
    print(f"[GraphRAG] 已缓存 embeddings 到：{EMBED_CACHE_PATH}")
    return embeds


def graph_rag_search(
    query_text: str,
    query_entities: List[str],
    corpus: List[CommunityDoc],
    corpus_embeds: np.ndarray,
    embed_model: SentenceTransformer,
    topk: int = 5,
    use_entity_prefilter: bool = True
) -> List[Dict[str, Any]]:
    """
    GraphRAG 检索：先 entity overlap 过滤（更接近 GraphRAG 思想），再用 summary embedding 排序
    """
    indices = list(range(len(corpus)))
    qset = set(query_entities)

    if use_entity_prefilter:
        filt = [i for i in indices if qset.intersection(corpus[i].entities)]
        if DEBUG:
            print(f"[DEBUG] entity-prefilter: {len(indices)} -> {len(filt)}")
        if filt:
            indices = filt

    q_emb = embed_model.encode([query_text], normalize_embeddings=True)[0]

    sims = np.dot(corpus_embeds[indices], q_emb)
    top_local = np.argsort(sims)[::-1][:topk]

    results = []
    for rank, local_idx in enumerate(top_local, 1):
        global_idx = indices[local_idx]
        c = corpus[global_idx]
        results.append({
            "rank": rank,
            "score": float(sims[local_idx]),
            "summary": c.summary,
            "entities": c.entities,
            "herbs": c.herbs
        })
    return results


def build_rag_context(results: List[Dict[str, Any]], herb_show_k: int = 15) -> str:
    blocks = []
    for r in results:
        herbs = r["herbs"][:herb_show_k]
        blocks.append(
            f"【社区 {r['rank']} | 相似度={r['score']:.3f}】\n"
            f"社区总结：\n{r['summary']}\n\n"
            f"该社区历史高频中药（Top{len(herbs)}）：\n{', '.join(herbs) if herbs else '（无统计到）'}"
        )
    return "\n\n".join(blocks)


# ======================================
# ======================================
SYSTEM_PROMPT = "你是一名经验丰富的中医辨证论治医生。"

USER_PROMPT_TEMPLATE = """
下面是当前病例的中医【判断线索】：
{features}

【相关知识社区（社区总结 + 历史高频中药）】
{rag_context}

【中药使用约束】
你只能从下面给定的【中药全集】中选择药物：
{HERB_VOCAB_TEXT}

任务：
1. 生成一个合理的中药处方，给出10种中药；
2. 最后一行必须严格输出：
中药：药1, 药2, 药3, 药4, 药5, 药6, 药7, 药8, 药9, 药10
"""


# ======================================
# ======================================
def debug_check_herb_coverage(truth_herbs: List[str], retrieved: List[Dict[str, Any]]) -> None:
    truth_set = set(truth_herbs)
    hit_any = False

    for r in retrieved:
        overlap = truth_set.intersection(r["herbs"])
        if overlap:
            print(f"[DEBUG] ✅ 社区 {r['rank']} 的社区中药命中真实：{sorted(list(overlap))}")
            hit_any = True

    if not hit_any:
        print("[DEBUG] ❌ 检索到的社区中药未命中任何真实中药（说明 RAG 阶段可能拿不到药）")


# ======================================
# ======================================
def run_experiment(
    test_path: str,
    train_path: str,
    max_cases: int = 100,
    force_rebuild_corpus: bool = False,
    force_rebuild_embeds: bool = False
):
    embed_model = SentenceTransformer("moka-ai/m3e-base")
    herb_to_category, categories = load_category_index(HERB_CATEGORY_DIR)
    cache = load_category_cache(CATEGORY_CACHE_PATH, categories)
    herb_to_category.update(cache)

    fallback = DEFAULT_CATEGORY if DEFAULT_CATEGORY in categories else categories[0]

    graph_corpus = build_graph_rag_corpus(train_path, force_rebuild=force_rebuild_corpus)

    corpus_embeds = build_or_load_embeddings(graph_corpus, embed_model, force_rebuild=force_rebuild_embeds)

    p10_scores = []
    acc_cl_scores = []
    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_cases:
                break

            row = json.loads(line)
            structured = row["structured"]

            truth_herbs = parse_truth_herbs(structured.get("推荐中药", ""))
            if not truth_herbs:
                if DEBUG:
                    print(f"[DEBUG] 样本 {idx+1} 无真实中药，跳过")
                continue

            query_structured = {k: v for k, v in structured.items() if k!= "总结描述" and k!= "推荐中药"}
            query_text = build_query_text(query_structured)
            query_entities = extract_entities_from_structured(query_structured)

            if DEBUG:
                print("\n" + "=" * 90)
                print(f"[DEBUG] 样本 {idx+1}")
                print("[DEBUG] Query Text：")
                print(query_text[:500])
                print("[DEBUG] Query Entities：", query_entities[:50])
                print("[DEBUG] Truth Herbs：", truth_herbs)

            rag_results = graph_rag_search(
                query_text=query_text,
                query_entities=query_entities,
                corpus=graph_corpus,
                corpus_embeds=corpus_embeds,
                embed_model=embed_model,
                topk=TOPK_RAG,
                use_entity_prefilter=True
            )

            if DEBUG:
                print("\n[DEBUG] ===== 检索到的社区 TopK =====")
                for r in rag_results:
                    print(f"\n--- 社区 {r['rank']} | sim={r['score']:.3f} ---")
                    print("社区中药Top10：", r["herbs"][:10])
                    print("社区总结(前200字)：", r["summary"][:200].replace("\n", " "))
                debug_check_herb_coverage(truth_herbs, rag_results)

            rag_context = build_rag_context(rag_results, herb_show_k=15)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                features=json.dumps(query_structured, ensure_ascii=False, indent=2),
                rag_context=rag_context,
                HERB_VOCAB_TEXT=HERB_VOCAB_TEXT
            )

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.001,
                "max_tokens": 1000
            }

            r = requests.post(GPT_API_URL, headers=HEADERS, data=json.dumps(payload), timeout=500)
            r.raise_for_status()
            output = r.json()["choices"][0]["message"]["content"].strip()

            pred_herbs = extract_herbs_from_output(output, TOPK_HERB)
            p10 = p_at_k(pred_herbs, set(truth_herbs), TOPK_HERB)
            p10_scores.append(p10)
            # ===== Acc-CL@10 =====
            pred_cats = herbs_to_categories(
                pred_herbs,
                herb_to_category,
                categories,
                CATEGORY_CACHE_PATH,
                fallback
            )

            truth_cats = herbs_to_categories(
                truth_herbs,
                herb_to_category,
                categories,
                CATEGORY_CACHE_PATH,
                fallback
            )

            p_cl, r_cl, f1_cl = f1_cl_score(pred_cats, truth_cats)

            acc_cl_scores.append(f1_cl)

            if DEBUG:
                print("预测中药大类：", pred_cats)
                print("真实中药大类：", truth_cats)
                print(f"Acc-CL@10 = {f1_cl:.3f}")

            print("\n【ChatGPT 输出】")
            print(output)
            print("预测 Top-10：", pred_herbs)
            print("真实中药：", truth_herbs)
            print("P@10 =", f"{p10:.3f}")
            print(f"CL-Precision = {p_cl:.3f}")
            print(f"CL-Recall    = {r_cl:.3f}")
            print(f"CL-F1        = {f1_cl:.3f}")

    print("\n====== GraphRAG(+Community Herbs) 最终结果 ======")
    if p10_scores:
        print(f"平均 P@10 = {float(np.mean(p10_scores)):.4f}")
    else:
        print("平均 P@10 = 0.0000")

    if acc_cl_scores:
        print(f"平均 Acc-CL@10 = {float(np.mean(acc_cl_scores)):.4f}")
    else:
        print("平均 Acc-CL@10 = 0.0000")


# ======================================
# ======================================
if __name__ == "__main__":
    run_experiment(
        test_path="test_jia_structured.jsonl",
        train_path="train_jia_structured.jsonl",
        max_cases=MAX_TEST_CASE,
        force_rebuild_corpus=True,
        force_rebuild_embeds=True
    )
