"""
TCM-HEDPR: Hierarchical Structure-Enhanced Personalized Recommendation
for Traditional Chinese Medicine Formulas (CIKM '25)

Inference-only adaptation for our dataset format.
No training loop — approximates all 5 modules via:
  - PEPP  : TF-IDF embedding of personalized patient prompt sequences
  - DMSH  : Cosine-similarity retrieval in "diffused" symptom-herb space (KG-augmented)
  - SYN   : LLM-based syndrome-aware prediction (GPT-4)
  - HGSN  : Herb-compatibility re-ranking via monarch/minister/assistant categorisation
  - PR    : Final scored herb list with top-K output

Data format expected (JSONL):
  {"id": "...", "metadata": {
      "现病史": "...", "四诊(规范)": "...",
      "病性(泛化)": "心阴虚,肝郁", "病位(泛化)": "心,肝",
      "中医辨证": "...", "中医诊断": "...", "西医诊断": "...",
      "中药名称": "麦冬,酸枣仁,..."
  }}

API: https://gpt-api.hkust-gz.edu.cn/v1/chat/completions  model=gpt-4
     Authorization: <bare key, no "Bearer">
"""

import json
import os
import re
import math
import time
import argparse
import collections
import requests
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_URL   = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
API_KEY   = "7bc70a62f11a48c18b00284cac02a7305753c7e4cce748bdb6d80791cfd32459"
MODEL     = "gpt-4"
TOP_K     = 10          # evaluate at K=10 (P@10, R@10, F1@10)
MAX_SIMILAR = 5         # similar patients retrieved for context
DIFFUSION_ALPHA = 0.4   # weight for original embedding (vs KG-augmented)
ALPHA_HPR = 0.4         # α in eq.18  (syndrome stream weight)
BETA_HPR  = 0.6         # β in eq.18  (herb-compat stream weight)
RETRY_DELAY = 5         # seconds between API retries
MAX_RETRIES = 3

# ─────────────────────────────────────────────
# HERB CATEGORY LOADING  (for Acc-CL metric)
# ─────────────────────────────────────────────
def load_herb_categories(cat_dir: str) -> Dict[str, str]:
    """Returns {herb_name: category_name}"""
    herb2cat = {}
    if not os.path.isdir(cat_dir):
        return herb2cat
    for fname in os.listdir(cat_dir):
        if fname.endswith(".txt"):
            cat = fname[:-4]
            with open(os.path.join(cat_dir, fname), encoding="utf-8") as f:
                for line in f:
                    h = line.strip()
                    if h:
                        herb2cat[h] = cat
    return herb2cat


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_herbs(meta: dict) -> List[str]:
    raw = meta.get("中药名称", "")
    if not raw:
        return []
    return [h.strip() for h in re.split(r"[,，、\s]+", raw) if h.strip()]


def build_patient_text(meta: dict, include_herbs: bool = True) -> str:
    """Build a unified text representation for PEPP module."""
    parts = []
    # Personalized patient attributes (PEPP prompt sequence)
    for key in ["现病史", "四诊(规范)", "病位(泛化)", "病性(泛化)",
                "中医辨证", "中医诊断", "西医诊断"]:
        val = meta.get(key, "")
        if val:
            parts.append(f"{key}: {val}")
    if include_herbs:
        herbs = parse_herbs(meta)
        if herbs:
            parts.append(f"中药: {' '.join(herbs)}")
    return " | ".join(parts)


def build_symptom_text(meta: dict) -> str:
    """Symptom-only text for the DMSH module."""
    parts = []
    for key in ["现病史", "四诊(规范)", "病位(泛化)", "病性(泛化)"]:
        val = meta.get(key, "")
        if val:
            parts.append(val)
    return " ".join(parts)


# ─────────────────────────────────────────────
# MODULE 1 – PEPP
# Patient-personalized pre-embedding via TF-IDF
# (replaces SASRec + MLP prompt generator)
# ─────────────────────────────────────────────
class PEPP:
    """
    Pre-embed patient personalised prompt sequences.
    Uses TF-IDF on full patient text (including herb history from train).
    Contrastive Learning approximated by mean-pooling augmented views.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            max_features=8192,
            sublinear_tf=True
        )
        self.train_embs: Optional[np.ndarray] = None
        self.train_texts: List[str] = []

    def fit(self, train_records: List[dict]):
        self.train_texts = [build_patient_text(r["metadata"], include_herbs=True)
                            for r in train_records]
        embs = self.vectorizer.fit_transform(self.train_texts).toarray().astype(np.float32)
        # CL data augmentation: average original + two masked views (feature dropout)
        rng = np.random.default_rng(42)
        mask1 = rng.random(embs.shape) > 0.2   # γ1=0.2 (mask patient attr)
        mask2 = rng.random(embs.shape) > 0.2   # γ2=0.2 (mask herb seq)
        aug1  = embs * mask1
        aug2  = embs * mask2
        embs_cl = (embs + aug1 + aug2) / 3.0
        self.train_embs = normalize(embs_cl)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return normalize(embs)

    def retrieve_similar(self, query_text: str, train_records: List[dict],
                         k: int = MAX_SIMILAR) -> List[dict]:
        q_emb = self.encode([query_text])
        sims  = cosine_similarity(q_emb, self.train_embs)[0]
        top_k = np.argsort(sims)[::-1][:k]
        return [(train_records[i], float(sims[i])) for i in top_k]


# ─────────────────────────────────────────────
# MODULE 2 – DMSH
# KG-guided symptom-herb diffusion
# Approximated as: herb co-occurrence weighted by KG similarity
# ─────────────────────────────────────────────
class DMSH:
    """
    Diffusion-guided Symptom-Herb representation learning.

    We approximate the forward/reverse diffusion process as:
      1. Build herb co-occurrence matrix from train (H-H homogeneous graph)
      2. Build symptom–herb frequency matrix (S-H bipartite)
      3. "KG diffusion": enrich herb embeddings with neighbour averaging
         simulating T diffusion steps via graph smoothing (eq. 5–10)
      4. Final herb score = α * direct_score + (1-α) * diffused_score
    """
    def __init__(self, diffusion_steps: int = 3, alpha: float = DIFFUSION_ALPHA):
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.herb_list: List[str] = []
        self.symptom_list: List[str] = []
        self.herb_idx: Dict[str, int] = {}
        self.symptom_idx: Dict[str, int] = {}
        # Matrices
        self.SH: Optional[np.ndarray] = None     # symptom × herb
        self.HH: Optional[np.ndarray] = None     # herb × herb (co-occurrence)
        self.SH_diffused: Optional[np.ndarray] = None

    def _tokenize_symptoms(self, meta: dict) -> List[str]:
        """Tokenise symptom text into tokens (chars + words for Chinese)."""
        text = build_symptom_text(meta)
        # Split on delimiters, keep CJK chars as individual tokens
        tokens = re.split(r"[,，、；;。\s|:：]+", text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens

    def fit(self, train_records: List[dict]):
        # Collect vocabulary
        all_symptoms_raw: List[List[str]] = []
        all_herbs_raw:    List[List[str]] = []
        for r in train_records:
            meta = r["metadata"]
            symptoms = self._tokenize_symptoms(meta)
            herbs    = parse_herbs(meta)
            all_symptoms_raw.append(symptoms)
            all_herbs_raw.append(herbs)

        # Build index
        all_sym_flat  = [s for lst in all_symptoms_raw for s in lst]
        all_herb_flat = [h for lst in all_herbs_raw    for h in lst]
        sym_counter  = collections.Counter(all_sym_flat)
        herb_counter = collections.Counter(all_herb_flat)
        # Filter low-frequency (like 10-kernel in paper)
        self.symptom_list = [s for s, c in sym_counter.items() if c >= 3]
        self.herb_list    = [h for h, c in herb_counter.items() if c >= 3]
        self.symptom_idx  = {s: i for i, s in enumerate(self.symptom_list)}
        self.herb_idx     = {h: i for i, h in enumerate(self.herb_list)}
        ns, nh = len(self.symptom_list), len(self.herb_list)
        if ns == 0 or nh == 0:
            return self

        # Build S-H matrix (symptom × herb) — counts
        SH = np.zeros((ns, nh), dtype=np.float32)
        for symptoms, herbs in zip(all_symptoms_raw, all_herbs_raw):
            for s in symptoms:
                if s in self.symptom_idx:
                    si = self.symptom_idx[s]
                    for h in herbs:
                        if h in self.herb_idx:
                            SH[si, self.herb_idx[h]] += 1.0

        # TF-IDF style normalisation on SH
        # IDF per herb
        doc_freq = (SH > 0).sum(axis=0) + 1.0
        idf = np.log((ns + 1.0) / doc_freq) + 1.0
        SH = SH * idf[None, :]
        self.SH = normalize(SH, norm="l2")  # ns × nh

        # Build H-H co-occurrence (homogeneous herb graph)
        HH = np.zeros((nh, nh), dtype=np.float32)
        for herbs in all_herbs_raw:
            idxs = [self.herb_idx[h] for h in herbs if h in self.herb_idx]
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    HH[idxs[i], idxs[j]] += 1.0
                    HH[idxs[j], idxs[i]] += 1.0
        # Row-normalise adjacency (like GCN)
        row_sum = HH.sum(axis=1, keepdims=True) + 1e-8
        HH_norm = HH / row_sum
        self.HH = HH_norm  # nh × nh

        # KG Diffusion (eq. 5-10 approximation):
        # Simulate T steps of graph diffusion on herb embeddings.
        # herb_emb: nh × nh  (herbs viewed through their HH-graph neighbourhood)
        # Initialise herb_emb as identity-like: each herb = its HH row
        herb_emb = self.HH.copy()  # nh × nh
        for _ in range(self.diffusion_steps):
            herb_emb = self.alpha * self.HH + (1 - self.alpha) * (self.HH @ herb_emb)
        herb_emb = normalize(herb_emb, norm="l2")  # nh × nh
        # Diffused S-H score matrix: ns × nh
        # SH_diffused[s, h] = SH[s, :] · herb_emb[h, :]^T  = SH @ herb_emb.T
        # herb_emb is nh×nh → herb_emb.T is nh×nh → SH(ns×nh) @ herb_emb.T(nh×nh) = ns×nh ✓
        self.SH_diffused = self.SH @ herb_emb.T  # ns × nh
        self.herb_emb    = herb_emb               # nh × nh (kept for reference)
        return self

    def score_herbs(self, meta: dict) -> Dict[str, float]:
        """Score all known herbs for a given patient's symptoms."""
        if self.SH is None:
            return {}
        symptoms = self._tokenize_symptoms(meta)
        ns = len(self.symptom_list)
        # Build query symptom vector
        q = np.zeros(ns, dtype=np.float32)
        for s in symptoms:
            if s in self.symptom_idx:
                q[self.symptom_idx[s]] += 1.0
        if q.sum() == 0:
            return {}
        q = q / (np.linalg.norm(q) + 1e-8)
        # Direct S-H score: q(ns,) @ SH(ns,nh) → (nh,)
        direct   = self.SH.T @ q              # nh
        # Diffused score: q(ns,) @ SH_diffused(ns,nh) → (nh,)
        diffused = self.SH_diffused.T @ q     # nh
        combined = self.alpha * direct + (1 - self.alpha) * diffused
        scores = {self.herb_list[i]: float(combined[i]) for i in range(len(self.herb_list))}
        return scores


# ─────────────────────────────────────────────
# MODULE 3 – SYN  (LLM-based syndrome-aware prediction)
# ─────────────────────────────────────────────
def call_llm(messages: list, temperature: float = 0.2) -> str:
    headers = {
        "Content-Type":  "application/json",
        "Authorization": API_KEY,
    }
    payload = {
        "model":       MODEL,
        "messages":    messages,
        "max_tokens":  1000,
        "temperature": temperature,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=500)
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [API] Retry {attempt+1} after error: {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


class SYN:
    """
    Syndrome-Aware Prediction (SYN module).
    Uses GPT-4 to:
      1. Infer the syndrome (辨证) from patient data + similar cases
      2. Recommend herbs based on syndrome + DMSH candidate herbs
    The prompt embeds personalised patient info (PEPP) + similar patients (DMSH).
    Herb compatibility (HGSN) is enforced via the system prompt.
    """

    SYSTEM_PROMPT = """你是一位资深中医临床医师，精通辨证论治。
请根据患者信息和相似病例进行以下操作：
1. 辨别主要证型（syndrome）
2. 按照"君臣佐使"配伍原则，从候选中药中推荐最适合的{top_k}味中药
要求：
- 严格基于中医理论进行辨证
- 优先考虑候选中药列表中的药物
- 君药解主证，臣药辅君，佐使协调
- 仅返回JSON，格式：{{"syndrome": "证型名称", "herbs": ["药1","药2",...]}}
- herbs列表恰好包含{top_k}味药，按君臣佐使顺序排列"""

    def predict(
        self,
        meta: dict,
        similar_cases: List[Tuple[dict, float]],
        candidate_herbs: List[str],
        top_k: int = TOP_K,
    ) -> Tuple[str, List[str]]:
        """Returns (syndrome_str, herb_list)."""

        # Build patient description
        patient_info = []
        for key in ["现病史", "四诊(规范)", "病位(泛化)", "病性(泛化)",
                    "中医辨证", "中医诊断", "西医诊断"]:
            val = meta.get(key, "")
            if val:
                patient_info.append(f"【{key}】{val}")
        patient_str = "\n".join(patient_info)

        # Build similar cases context
        similar_str = ""
        if similar_cases:
            sc_parts = []
            for i, (sim_meta, sim_score) in enumerate(similar_cases[:3], 1):
                sc_herbs = parse_herbs(sim_meta)
                sc_syn   = sim_meta.get("中医辨证", "")
                sc_sym   = sim_meta.get("四诊(规范)", "")[:80]
                sc_parts.append(
                    f"相似病例{i}(相似度{sim_score:.2f})：症状={sc_sym}；"
                    f"辨证={sc_syn}；处方={'、'.join(sc_herbs[:8])}"
                )
            similar_str = "\n".join(sc_parts)

        # Candidate herbs (top scored by DMSH)
        cand_str = "、".join(candidate_herbs[:30]) if candidate_herbs else "无候选"

        user_msg = f"""患者信息：
{patient_str}

参考相似病例：
{similar_str if similar_str else "无"}

DMSH模块候选中药（按相关度排序）：
{cand_str}

请进行辨证并从候选中药中选取{top_k}味组成处方，严格按JSON格式返回。"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(top_k=top_k)},
            {"role": "user",   "content": user_msg},
        ]
        raw = call_llm(messages)
        return self._parse_response(raw, candidate_herbs, top_k)

    @staticmethod
    def _parse_response(raw: str, fallback_herbs: List[str],
                        top_k: int) -> Tuple[str, List[str]]:
        # Extract JSON from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                syndrome = obj.get("syndrome", "")
                herbs    = obj.get("herbs", [])
                if isinstance(herbs, list) and herbs:
                    return syndrome, [str(h).strip() for h in herbs[:top_k]]
            except json.JSONDecodeError:
                pass
        # Fallback: return top DMSH herbs
        return "", fallback_herbs[:top_k]


# ─────────────────────────────────────────────
# MODULE 4 – HGSN
# Herb compatibility re-ranking (monarch/minister/assistant)
# Approximated via co-occurrence statistics from train
# ─────────────────────────────────────────────
class HGSN:
    """
    Heterogeneous Graph-based Hierarchical Structured Network for herbs.
    Classifies herbs into monarch/minister/assistant & envoy roles
    using frequency-based heuristics (eq. 13-16 approximation).
    """
    def __init__(self):
        self.herb_freq: Dict[str, int] = {}   # overall frequency
        self.herb_lead: Dict[str, int] = {}   # times herb appears first (monarch proxy)
        self.herb_cooc: Dict[Tuple[str,str], int] = {}  # pairwise co-occurrence

    def fit(self, train_records: List[dict]):
        for r in train_records:
            herbs = parse_herbs(r["metadata"])
            for i, h in enumerate(herbs):
                self.herb_freq[h] = self.herb_freq.get(h, 0) + 1
                if i == 0:
                    self.herb_lead[h] = self.herb_lead.get(h, 0) + 1
            # pairwise
            for i in range(len(herbs)):
                for j in range(i+1, len(herbs)):
                    key = (herbs[i], herbs[j])
                    self.herb_cooc[key] = self.herb_cooc.get(key, 0) + 1
        return self

    def herb_role_score(self, herb: str) -> Tuple[float, float, float]:
        """Returns (monarch_score, minister_score, asst_score)."""
        freq = self.herb_freq.get(herb, 0)
        lead = self.herb_lead.get(herb, 0)
        total_cases = max(sum(1 for v in self.herb_freq.values()), 1)
        if freq == 0:
            return 0.0, 0.0, 0.0
        monarch_score  = lead / (freq + 1e-8)
        minister_score = (freq - lead) / (total_cases + 1e-8)
        asst_score     = 1.0 / (freq + 1.0)   # rarer herbs tend to be assistant
        return monarch_score, minister_score, asst_score

    def rerank(self, candidate_herbs: List[str], syndrome_herbs: List[str],
               dmsh_scores: Dict[str, float], top_k: int = TOP_K) -> List[str]:
        """
        Combine DMSH scores + SYN syndrome herbs + HGSN compatibility.
        eq. 18: ŷ = α * M_pf + β * (M_C + M_D + M_AE)
        """
        all_herbs = list(dict.fromkeys(syndrome_herbs + candidate_herbs))
        final_scores = {}
        for h in all_herbs:
            # α * syndrome stream (SYN module output — binary indicator)
            syn_score = ALPHA_HPR * (1.0 if h in syndrome_herbs else 0.0)
            # β * herb compatibility stream (HGSN)
            dmsh_raw = dmsh_scores.get(h, 0.0)
            mc, md, mae = self.herb_role_score(h)
            # monarch + minister + assistant streams (eq.15)
            compat_score = BETA_HPR * (dmsh_raw + mc + md + mae) / 4.0
            final_scores[h] = syn_score + compat_score

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in ranked[:top_k]]


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(predictions: List[List[str]],
             ground_truths: List[List[str]],
             herb2cat: Dict[str, str]) -> dict:
    p_list, r_list, f1_list = [], [], []
    f1_cl_list = []

    for pred, gt in zip(predictions, ground_truths):
        pred_set = set(pred)
        gt_set   = set(gt)
        if not gt_set:
            continue
        hit = len(pred_set & gt_set)
        p   = hit / len(pred_set) if pred_set else 0.0
        r   = hit / len(gt_set)
        f1  = 2 * p * r / (p + r + 1e-8)
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)

        # Category-level F1 (Acc-CL)
        if herb2cat:
            pred_cats = set(herb2cat.get(h, h) for h in pred_set)
            gt_cats   = set(herb2cat.get(h, h) for h in gt_set)
            hit_cl = len(pred_cats & gt_cats)
            p_cl   = hit_cl / len(pred_cats) if pred_cats else 0.0
            r_cl   = hit_cl / len(gt_cats)   if gt_cats   else 0.0
            f1_cl  = 2 * p_cl * r_cl / (p_cl + r_cl + 1e-8)
            f1_cl_list.append(f1_cl)

    n = len(p_list)
    results = {
        f"P@{TOP_K}":    round(sum(p_list)  / n, 4) if n else 0,
        f"R@{TOP_K}":    round(sum(r_list)  / n, 4) if n else 0,
        f"F1@{TOP_K}":   round(sum(f1_list) / n, 4) if n else 0,
    }
    if f1_cl_list:
        results[f"Acc-CL@{TOP_K}"] = round(sum(f1_cl_list) / len(f1_cl_list), 4)
    return results


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
class TCM_HEDPR:
    def __init__(self):
        self.pepp = PEPP()
        self.dmsh = DMSH()
        self.syn  = SYN()
        self.hgsn = HGSN()
        self.train_records: List[dict] = []

    # ── Step 1: Build index from train ──
    def fit(self, train_records: List[dict]):
        print(f"[HEDPR] Fitting on {len(train_records)} train records …")
        self.train_records = train_records
        self.pepp.fit(train_records)
        print("  [PEPP] TF-IDF embeddings built")
        self.dmsh.fit(train_records)
        print(f"  [DMSH] Herb vocab={len(self.dmsh.herb_list)}, "
              f"Symptom vocab={len(self.dmsh.symptom_list)}")
        self.hgsn.fit(train_records)
        print(f"  [HGSN] Herb frequency table built "
              f"({len(self.hgsn.herb_freq)} herbs)")
        return self

    # ── Step 2: Predict for a single patient ──
    def predict_one(self, meta: dict, verbose: bool = False) -> List[str]:
        # PEPP: retrieve similar patients
        q_text = build_patient_text(meta, include_herbs=False)
        similar = self.pepp.retrieve_similar(q_text, self.train_records, k=MAX_SIMILAR)
        similar_metas = [(r["metadata"], s) for r, s in similar]

        # DMSH: score herbs by symptom-herb diffusion
        dmsh_scores = self.dmsh.score_herbs(meta)
        # Sort candidates by DMSH score
        cand_herbs = sorted(dmsh_scores, key=lambda h: dmsh_scores[h], reverse=True)

        # SYN: LLM syndrome-aware prediction
        syndrome, syn_herbs = self.syn.predict(
            meta=meta,
            similar_cases=similar_metas,
            candidate_herbs=cand_herbs,
            top_k=TOP_K,
        )
        if verbose:
            print(f"    Syndrome: {syndrome}")
            print(f"    SYN herbs: {syn_herbs}")

        # HGSN + PR: final re-ranking
        final_herbs = self.hgsn.rerank(
            candidate_herbs=cand_herbs,
            syndrome_herbs=syn_herbs,
            dmsh_scores=dmsh_scores,
            top_k=TOP_K,
        )
        return final_herbs

    # ── Step 3: Evaluate on test set ──
    def evaluate(self, test_records: List[dict],
                 herb2cat: Dict[str, str],
                 output_file: Optional[str] = None,
                 verbose: bool = False) -> dict:

        predictions  = []
        ground_truths = []
        output_rows  = []

        for i, r in enumerate(test_records):
            meta = r["metadata"]
            gt   = parse_herbs(meta)
            print(f"[{i+1}/{len(test_records)}] id={r.get('id','?')} "
                  f"gt_herbs={len(gt)}")
            try:
                pred = self.predict_one(meta, verbose=verbose)
            except Exception as e:
                print(f"  [ERROR] {e}  — using empty prediction")
                pred = []
            predictions.append(pred)
            ground_truths.append(gt)
            output_rows.append({
                "id":         r.get("id", str(i)),
                "prediction": pred,
                "ground_truth": gt,
            })
            # Save incrementally
            if output_file:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_rows[-1], ensure_ascii=False) + "\n")

        results = evaluate(predictions, ground_truths, herb2cat)
        return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TCM-HEDPR inference pipeline")
    parser.add_argument("--train",      default="train.jsonl")
    parser.add_argument("--test",       default="test.jsonl")
    parser.add_argument("--cat_dir",    default="herb_category",
                        help="Dir with category .txt files for Acc-CL metric")
    parser.add_argument("--output",     default="predictions.jsonl",
                        help="Output file for predictions")
    parser.add_argument("--max_test",   type=int, default=None,
                        help="Limit number of test samples (for quick debug)")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    # Load data
    train_records = load_jsonl(args.train)
    test_records  = load_jsonl(args.test)
    if args.max_test:
        test_records = test_records[:args.max_test]
    herb2cat = load_herb_categories(args.cat_dir)
    print(f"Train: {len(train_records)}, Test: {len(test_records)}, "
          f"Herb categories: {len(set(herb2cat.values()))}")

    # Clear output file
    if os.path.exists(args.output):
        os.remove(args.output)

    # Run
    model = TCM_HEDPR()
    model.fit(train_records)
    results = model.evaluate(
        test_records, herb2cat,
        output_file=args.output,
        verbose=args.verbose,
    )

    print("\n" + "="*50)
    print("TCM-HEDPR Evaluation Results")
    print("="*50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("="*50)

    # Save summary
    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
