"""
Ablation study for TCM-HEDPR (Table 2 variants):
  - full      : TCM-HEDPR (all modules)
  - w/o PEPP  : skip similar-patient retrieval, use empty context
  - w/o DMSH  : replace diffusion scores with random baseline
  - w/o SYN   : skip LLM call, use DMSH+HGSN only
  - w/o HGSN  : skip herb-compatibility re-ranking
  - w/o IKG   : skip KG diffusion (α=1, no graph smoothing)

Usage:
  python ablation.py --train train.jsonl --test test.jsonl \
                     --cat_dir herb_category --max_test 50
"""
import json
import os
import argparse
import copy
import numpy as np
from tcm_hedpr import (
    TCM_HEDPR, PEPP, DMSH, SYN, HGSN,
    load_jsonl, parse_herbs, load_herb_categories,
    build_patient_text, evaluate, TOP_K, MAX_SIMILAR, ALPHA_HPR, BETA_HPR
)


# ── Variant runners ──────────────────────────────────────────────────────────

def predict_full(model: TCM_HEDPR, meta: dict) -> list:
    return model.predict_one(meta)


def predict_no_pepp(model: TCM_HEDPR, meta: dict) -> list:
    """w/o PEPP: no similar-patient retrieval → empty context for SYN."""
    from tcm_hedpr import build_symptom_text, TOP_K
    dmsh_scores = model.dmsh.score_herbs(meta)
    cand_herbs  = sorted(dmsh_scores, key=lambda h: dmsh_scores[h], reverse=True)
    # SYN with no similar cases
    syndrome, syn_herbs = model.syn.predict(
        meta=meta, similar_cases=[], candidate_herbs=cand_herbs, top_k=TOP_K
    )
    return model.hgsn.rerank(cand_herbs, syn_herbs, dmsh_scores, TOP_K)


def predict_no_dmsh(model: TCM_HEDPR, meta: dict) -> list:
    """w/o DMSH: replace diffusion scores with uniform (like VAE baseline)."""
    q_text = build_patient_text(meta, include_herbs=False)
    similar = model.pepp.retrieve_similar(q_text, model.train_records, k=MAX_SIMILAR)
    similar_metas = [(r["metadata"], s) for r, s in similar]
    # uniform herb scores (simulate VAE generative model without structure)
    uniform_scores = {h: 1.0 for h in model.hgsn.herb_freq}
    cand_herbs = list(model.hgsn.herb_freq.keys())
    syndrome, syn_herbs = model.syn.predict(
        meta=meta, similar_cases=similar_metas,
        candidate_herbs=cand_herbs[:30], top_k=TOP_K
    )
    return model.hgsn.rerank(cand_herbs, syn_herbs, uniform_scores, TOP_K)


def predict_no_syn(model: TCM_HEDPR, meta: dict) -> list:
    """w/o SYN: skip LLM call; use DMSH + HGSN only."""
    dmsh_scores = model.dmsh.score_herbs(meta)
    cand_herbs  = sorted(dmsh_scores, key=lambda h: dmsh_scores[h], reverse=True)
    # HGSN rerank with empty syndrome herbs
    return model.hgsn.rerank(cand_herbs, [], dmsh_scores, TOP_K)


def predict_no_hgsn(model: TCM_HEDPR, meta: dict) -> list:
    """w/o HGSN: skip herb-compatibility; use SYN output directly."""
    q_text  = build_patient_text(meta, include_herbs=False)
    similar = model.pepp.retrieve_similar(q_text, model.train_records, k=MAX_SIMILAR)
    similar_metas = [(r["metadata"], s) for r, s in similar]
    dmsh_scores   = model.dmsh.score_herbs(meta)
    cand_herbs    = sorted(dmsh_scores, key=lambda h: dmsh_scores[h], reverse=True)
    _, syn_herbs  = model.syn.predict(
        meta=meta, similar_cases=similar_metas,
        candidate_herbs=cand_herbs, top_k=TOP_K
    )
    # Fall back to DMSH order for missing herbs
    all_herbs = list(dict.fromkeys(syn_herbs + cand_herbs))
    return all_herbs[:TOP_K]


def predict_no_ikg(model: TCM_HEDPR, meta: dict) -> list:
    """w/o IKG: set DMSH diffusion alpha=1.0 (no graph smoothing).
    We approximate by using the raw SH scores only (direct symptom-herb counts).
    """
    from tcm_hedpr import build_symptom_text
    import re as _re
    # Recompute score without diffusion
    dmsh = model.dmsh
    if dmsh.SH is None:
        return []
    # Tokenise symptoms
    text = build_symptom_text(meta)
    tokens = _re.split(r"[,，、；;。\s|:：]+", text)
    tokens = [t.strip() for t in tokens if t.strip()]
    ns = len(dmsh.symptom_list)
    q  = np.zeros(ns, dtype=np.float32)
    for s in tokens:
        if s in dmsh.symptom_idx:
            q[dmsh.symptom_idx[s]] += 1.0
    if q.sum() == 0:
        return []
    q /= (np.linalg.norm(q) + 1e-8)
    # Direct score only (α=1, no diffusion)
    direct = dmsh.SH.T @ q    # nh,  SH is ns×nh → SH.T is nh×ns → nh×ns @ ns = nh ✓
    scores = {dmsh.herb_list[i]: float(direct[i]) for i in range(len(dmsh.herb_list))}
    cand_herbs = sorted(scores, key=lambda h: scores[h], reverse=True)

    q_text  = build_patient_text(meta, include_herbs=False)
    similar = model.pepp.retrieve_similar(q_text, model.train_records, k=MAX_SIMILAR)
    similar_metas = [(r["metadata"], s) for r, s in similar]
    _, syn_herbs  = model.syn.predict(
        meta=meta, similar_cases=similar_metas,
        candidate_herbs=cand_herbs, top_k=TOP_K
    )
    return model.hgsn.rerank(cand_herbs, syn_herbs, scores, TOP_K)


VARIANTS = {
    "TCM-HEDPR (full)": predict_full,
    "w/o PEPP":         predict_no_pepp,
    "w/o DMSH":         predict_no_dmsh,
    "w/o SYN":          predict_no_syn,
    "w/o HGSN":         predict_no_hgsn,
    "w/o IKG":          predict_no_ikg,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--test",       required=True)
    parser.add_argument("--cat_dir",    default="herb_category")
    parser.add_argument("--max_test",   type=int, default=None)
    parser.add_argument("--variants",   nargs="+", default=list(VARIANTS.keys()),
                        help="Which variants to run (default: all)")
    parser.add_argument("--output",     default="ablation_results.json")
    args = parser.parse_args()

    train_records = load_jsonl(args.train)
    test_records  = load_jsonl(args.test)
    if args.max_test:
        test_records = test_records[:args.max_test]
    herb2cat = load_herb_categories(args.cat_dir)

    print(f"Train: {len(train_records)}, Test: {len(test_records)}")
    model = TCM_HEDPR()
    model.fit(train_records)

    all_results = {}
    for variant_name in args.variants:
        if variant_name not in VARIANTS:
            print(f"  [SKIP] Unknown variant: {variant_name}")
            continue
        pred_fn = VARIANTS[variant_name]
        print(f"\n{'='*50}")
        print(f"Running variant: {variant_name}")
        print(f"{'='*50}")

        predictions, ground_truths = [], []
        for i, r in enumerate(test_records):
            meta = r["metadata"]
            gt   = parse_herbs(meta)
            print(f"  [{i+1}/{len(test_records)}]", end=" ", flush=True)
            try:
                pred = pred_fn(model, meta)
            except Exception as e:
                print(f"ERROR: {e}")
                pred = []
            predictions.append(pred)
            ground_truths.append(gt)
            print(f"pred={len(pred)} gt={len(gt)}")

        result = evaluate(predictions, ground_truths, herb2cat)
        all_results[variant_name] = result
        print(f"  Result: {result}")

    # Summary table
    print("\n" + "="*70)
    print(f"{'Variant':<25} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'Acc-CL':>8}")
    print("="*70)
    for name, res in all_results.items():
        print(f"{name:<25} {res.get(f'P@{TOP_K}',0):>8.4f} "
              f"{res.get(f'R@{TOP_K}',0):>8.4f} "
              f"{res.get(f'F1@{TOP_K}',0):>8.4f} "
              f"{res.get(f'Acc-CL@{TOP_K}',0):>8.4f}")
    print("="*70)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
