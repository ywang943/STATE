# ======================================
# ======================================
import json
from collections import defaultdict
from typing import Dict, Set, List


# ======================================
# ======================================
class TCMHypergraph:
    """
    中医经验 Hypergraph（二部超图）
    - Entity <-> Hyperedge
    """

    def __init__(self):
        # entity -> set(edge_id)
        self.entity_to_edges: Dict[str, Set[str]] = defaultdict(set)

        # edge_id -> set(entity)
        self.edge_to_entities: Dict[str, Set[str]] = defaultdict(set)

        # edge_id -> hyperedge attributes
        self.edge_attrs: Dict[str, Dict] = {}


# ======================================
# ======================================
ENTITY_FIELDS = [
    "寒热",
    "虚实",
    "表里",
    "涉及可能脏腑",
    "涉及典型病机",
    "动态特征",
    "时间节律",
    "饮食相关",
    "情志相关",
    "消化表现"
]


# ======================================
# ======================================
def make_entity(field: str, value: str) -> str:
    """
    统一 entity 表示：字段=取值
    """
    return f"{field}={value}"


def extract_entities(structured: dict) -> List[str]:
    """
    从 structured case 中抽取实体
    """
    entities = []
    for field in ENTITY_FIELDS:
        value = structured.get(field, "")
        if value and value != "暂不确定":
            entities.append(make_entity(field, value))
    return entities


# ======================================
# ======================================
def add_hyperedge(
    hg: TCMHypergraph,
    edge_id: str,
    structured: dict
):
    """
    将一个 structured case 加入 Hypergraph
    """
    entities = extract_entities(structured)

    for ent in entities:
        hg.entity_to_edges[ent].add(edge_id)
        hg.edge_to_entities[edge_id].add(ent)

    hg.edge_attrs[edge_id] = {
        "总结描述": structured.get("总结描述", ""),
        "置信分数": float(structured.get("置信分数", "0.0")),
        "病位": structured.get("病位", ""),
        "病性": structured.get("病性", ""),
        "推荐中药": structured.get("推荐中药", ""),
        "entities": entities
    }


# ======================================
# ======================================
def build_hypergraph_from_jsonl(path: str) -> TCMHypergraph:
    hg = TCMHypergraph()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            edge_id = str(row["id"])
            structured = row["structured"]

            add_hyperedge(hg, edge_id, structured)

    return hg


# ======================================
# ======================================
def save_hyperedges_jsonl(hg: TCMHypergraph, path: str):
    """
    保存 hyperedge 主数据库（经验库）
    """
    with open(path, "w", encoding="utf-8") as f:
        for edge_id, attrs in hg.edge_attrs.items():
            record = {
                "edge_id": edge_id,
                "entities": sorted(list(hg.edge_to_entities[edge_id])),
                "总结描述": attrs.get("总结描述", ""),
                "置信分数": attrs.get("置信分数", 0.0),
                "病位": attrs.get("病位", ""),
                "病性": attrs.get("病性", ""),
                "推荐中药": attrs.get("推荐中药", "")
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_entity_index_jsonl(hg: TCMHypergraph, path: str):
    """
    保存 entity -> hyperedge 的倒排索引
    """
    with open(path, "w", encoding="utf-8") as f:
        for entity, edge_ids in hg.entity_to_edges.items():
            record = {
                "entity": entity,
                "edge_ids": sorted(list(edge_ids))
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ======================================
# ======================================
def retrieve_by_structure(
    hg: TCMHypergraph,
    query_entities: Set[str],
    min_hit: int = 2
):
    """
    基于 |Vq ∩ Ve| 的纯结构检索
    """
    results = []
    for edge_id, Ve in hg.edge_to_entities.items():
        hit = len(query_entities & Ve)
        if hit >= min_hit:
            results.append((edge_id, hit))
    return sorted(results, key=lambda x: x[1], reverse=True)


# ======================================
# ======================================
if __name__ == "__main__":
    INPUT_PATH = "train_jia_structured.jsonl"
    OUTPUT_HYPEREDGE_DB = "hyperedges.jsonl"
    OUTPUT_ENTITY_INDEX = "entity_index.jsonl"

    print("🚧 Step 1: 构建 Hypergraph ...")
    hg = build_hypergraph_from_jsonl(INPUT_PATH)

    print("✅ Hypergraph 构建完成")
    print(f"Hyperedge 数量: {len(hg.edge_to_entities)}")
    print(f"Entity 数量: {len(hg.entity_to_edges)}")

    print("\n💾 Step 2: 保存为 JSONL 数据库 ...")
    save_hyperedges_jsonl(hg, OUTPUT_HYPEREDGE_DB)
    save_entity_index_jsonl(hg, OUTPUT_ENTITY_INDEX)

    print("✅ 保存完成")
    print("生成文件：")
    print(f" - {OUTPUT_HYPEREDGE_DB}  （经验主库）")
    print(f" - {OUTPUT_ENTITY_INDEX}  （结构倒排索引）")

    # ======== Sanity Check ========
    print("\n🔍 Sanity Check 示例")

    sample_edge = next(iter(hg.edge_attrs))
    print("示例 Hyperedge ID:", sample_edge)
    print("实体:", hg.edge_to_entities[sample_edge])
    print("属性:", hg.edge_attrs[sample_edge])

    sample_query = set(list(hg.edge_to_entities[sample_edge])[:3])
    print("\n模拟查询实体:", sample_query)

    hits = retrieve_by_structure(hg, sample_query, min_hit=2)
    print("结构命中 Top-5:", hits[:5])
