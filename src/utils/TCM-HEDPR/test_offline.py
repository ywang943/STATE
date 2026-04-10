"""
Offline smoke test — validates all modules WITHOUT calling the API.
Creates synthetic data, runs PEPP + DMSH + HGSN, checks shapes.
SYN (LLM) is mocked.
"""
import json
import os
import sys
import unittest
import tempfile
import numpy as np

# Patch sys.path so we can import from sibling dir when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from tcm_hedpr import (
    PEPP, DMSH, HGSN, SYN,
    TCM_HEDPR, evaluate,
    load_jsonl, parse_herbs, build_patient_text,
    TOP_K
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

HERBS = ["麦冬", "酸枣仁", "黄芪", "当归", "白术", "茯苓",
         "党参", "甘草", "柴胡", "白芍", "川芎", "丹参"]

SYMPTOMS = ["胸闷", "心悸", "气短", "失眠", "头晕", "乏力"]

def make_record(idx: int) -> dict:
    rng = np.random.default_rng(idx)
    chosen_syms  = list(rng.choice(SYMPTOMS, size=3, replace=False))
    chosen_herbs = list(rng.choice(HERBS, size=6, replace=False))
    return {
        "id": str(idx),
        "metadata": {
            "现病史":   f"患者{idx}月余，" + "，".join(chosen_syms),
            "四诊(规范)": "、".join(chosen_syms),
            "病性(泛化)": "气虚,血瘀",
            "病位(泛化)": "心,脾",
            "中医辨证":  "气虚血瘀证",
            "中医诊断":  "心悸",
            "西医诊断":  "心律失常",
            "中药名称":  "，".join(chosen_herbs),
        }
    }

def make_dataset(n: int):
    return [make_record(i) for i in range(n)]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPEPP(unittest.TestCase):
    def setUp(self):
        self.train = make_dataset(50)
        self.test  = make_dataset(5)

    def test_fit_encode(self):
        pepp = PEPP()
        pepp.fit(self.train)
        self.assertIsNotNone(pepp.train_embs)
        self.assertEqual(pepp.train_embs.shape[0], 50)

    def test_retrieve_similar(self):
        pepp = PEPP()
        pepp.fit(self.train)
        q_text = build_patient_text(self.test[0]["metadata"], include_herbs=False)
        similar = pepp.retrieve_similar(q_text, self.train, k=5)
        self.assertEqual(len(similar), 5)
        for rec, score in similar:
            self.assertIn("metadata", rec)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0 + 1e-6)


class TestDMSH(unittest.TestCase):
    def setUp(self):
        self.train = make_dataset(80)

    def test_fit(self):
        dmsh = DMSH()
        dmsh.fit(self.train)
        self.assertGreater(len(dmsh.herb_list), 0)
        self.assertGreater(len(dmsh.symptom_list), 0)
        self.assertIsNotNone(dmsh.SH)

    def test_score_herbs(self):
        dmsh = DMSH()
        dmsh.fit(self.train)
        meta   = make_dataset(1)[0]["metadata"]
        scores = dmsh.score_herbs(meta)
        self.assertIsInstance(scores, dict)
        for h, s in scores.items():
            self.assertIsInstance(s, float)


class TestHGSN(unittest.TestCase):
    def setUp(self):
        self.train = make_dataset(80)

    def test_fit_and_rerank(self):
        hgsn = HGSN()
        hgsn.fit(self.train)
        self.assertGreater(len(hgsn.herb_freq), 0)
        cand  = HERBS[:8]
        dmsh_scores = {h: float(i) for i, h in enumerate(cand)}
        result = hgsn.rerank(cand, cand[:3], dmsh_scores, top_k=5)
        self.assertEqual(len(result), 5)
        for h in result:
            self.assertIn(h, HERBS)


class TestEvaluate(unittest.TestCase):
    def test_perfect(self):
        preds = [["a","b","c","d","e","f","g","h","i","j"]]
        gts   = [["a","b","c","d","e","f","g","h","i","j"]]
        r = evaluate(preds, gts, {})
        self.assertAlmostEqual(r[f"P@{TOP_K}"], 1.0)
        self.assertAlmostEqual(r[f"R@{TOP_K}"], 1.0)

    def test_zero(self):
        preds = [["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"]]
        gts   = [["a","b","c"]]
        r = evaluate(preds, gts, {})
        self.assertAlmostEqual(r[f"P@{TOP_K}"], 0.0)

    def test_partial(self):
        preds = [["a","b","x","y","z","1","2","3","4","5"]]
        gts   = [["a","b","c"]]
        r = evaluate(preds, gts, {})
        # precision = 2/10 = 0.2, recall = 2/3
        self.assertAlmostEqual(r[f"P@{TOP_K}"], 0.2, places=3)
        self.assertAlmostEqual(r[f"R@{TOP_K}"], 2/3, places=3)

    def test_category_metric(self):
        herb2cat = {"a": "补气", "b": "补气", "c": "活血", "x": "活血"}
        preds = [["x", "y","1","2","3","4","5","6","7","8"]]
        gts   = [["a", "c"]]
        r = evaluate(preds, gts, herb2cat)
        self.assertIn(f"Acc-CL@{TOP_K}", r)


class TestIntegration(unittest.TestCase):
    """Integration test with mocked LLM."""
    def test_pipeline_no_api(self):
        """Run full pipeline but mock the SYN.predict call."""
        import unittest.mock as mock

        train = make_dataset(60)
        test  = make_dataset(3)

        model = TCM_HEDPR()
        model.fit(train)

        # Mock SYN to avoid API call
        with mock.patch.object(model.syn, "predict",
                               return_value=("气虚血瘀证", HERBS[:TOP_K])):
            for r in test:
                pred = model.predict_one(r["metadata"])
                self.assertEqual(len(pred), TOP_K)
                for h in pred:
                    self.assertIsInstance(h, str)
                    self.assertGreater(len(h), 0)

        print("  Integration test passed.")


# ── JSONL I/O test ────────────────────────────────────────────────────────────

class TestJSONL(unittest.TestCase):
    def test_roundtrip(self):
        dataset = make_dataset(5)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                        encoding="utf-8", delete=False) as f:
            for r in dataset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            path = f.name
        loaded = load_jsonl(path)
        self.assertEqual(len(loaded), 5)
        for r in loaded:
            herbs = parse_herbs(r["metadata"])
            self.assertGreater(len(herbs), 0)
        os.unlink(path)


if __name__ == "__main__":
    print("Running TCM-HEDPR offline smoke tests …\n")
    unittest.main(verbosity=2)
