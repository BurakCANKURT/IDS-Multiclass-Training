# -*- coding: utf-8 -*-
"""
R50 Gate-0 -- sinif-kosullu CORAL / CDAN-lite mekanik kill-gate
(DG makalesi Tablo 6 ve Tablo 7). EGITIM YAPMAZ: diskteki `.r50_*` LSTM
kollarini ve `.r49v3_win` scaler'ini kullanir.

Soru: domain-invariant ceza, AYNI ETIKET altindaki iki domainin (CICIDS vs
selfgen-WIN bloku) temsil-uzayi ayrilabilirligini dusuruyor mu?

On-kayitli kriter: penultimate Dense(32) temsilinde domain-ayrim AUC <= 0.75.
Gecmezse downstream (GATE-1) hic kosulmaz.

Protokol
  * domain s = CICIDS egitim bolmesi (batch_level_split) satirlari, sinif c
  * domain t = selfgen_dataset.csv, batch 'selfgen_mixed_win_20260820_182324', sinif c
  * her tekrar: dengeli alt-ornek (min(n_s, n_t, N_MAX) / domain), 5-kat
    tabakali CV, out-of-fold AUC; LR (standartlastirilmis) ve RF
  * 20 bagimsiz tekrar -> ortalama ve %95 percentil araligi (Tablo 6)
  * Tablo 7: WebAttack temsilinde moment ayristirmasi
      ||mu_s - mu_t||, ||C_s - C_t||_F, en iyi tek-boyut AUC, o boyutta
      deger-araligi ortusmesi = |kesisim| / |birlesim| ([min,max] araliklari)
  * Kok neden: GIRDI uzayinda ozellik basina AUC + aralik ortusmesi

Kullanim: python analysis/killgates/r50_coral_gate0.py [--reps 20]
Cikti   : analysis/killgates/out/r50_coral_gate0.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WIN_BATCH = "selfgen_mixed_win_20260820_182324"
CLASSES = ["WebAttack", "FTP-Patator"]
ARMS = [("raw_input_41", None),
        ("erm_control", ".r50_ermwin"),
        ("coral_l0.1", ".r50_coral_l01"),
        ("coral_l1", ".r50_coral_l1"),
        ("coral_l100", ".r50_coral_l100"),
        ("cdan_l1", ".r50_cdan"),
        ("cdan_l10", ".r50_cdan10")]
N_MAX = 3000
CRITERION = 0.75
MEMO_TABLE7 = {"erm_control": [2.103, 7.065, 0.9315, 0.2518],
               "coral_l0.1": [1.713, 3.731, 0.9625, 0.1961],
               "coral_l1": [1.708, 2.118, 0.9657, 0.2188],
               "coral_l100": [1.588, 0.245, 1.0000, 0.0000]}


def penultimate(suffix):
    import tensorflow as tf
    m = tf.keras.models.load_model(common.model_paths(suffix)["lstm"], compile=False)
    dense32 = [l for l in m.layers if l.__class__.__name__ == "Dense"][0]
    assert dense32.units == 32, dense32.units
    return tf.keras.Model(inputs=m.inputs, outputs=dense32.output)


def overlap(a, b):
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    span = max(a.max(), b.max()) - min(a.min(), b.min())
    return float(max(0.0, hi - lo) / span) if span > 0 else 1.0


def domain_auc(Zs, Zt, reps):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    n = min(len(Zs), len(Zt), N_MAX)
    out = {"lr": [], "rf": []}
    for r in range(reps):
        rng = np.random.RandomState(common.SEED + r)
        X = np.vstack([Zs[rng.choice(len(Zs), n, replace=False)],
                       Zt[rng.choice(len(Zt), n, replace=False)]])
        y = np.r_[np.zeros(n), np.ones(n)]
        cv = StratifiedKFold(5, shuffle=True, random_state=common.SEED + r)
        for name, clf in (("lr", make_pipeline(StandardScaler(),
                                               LogisticRegression(max_iter=2000))),
                          ("rf", RandomForestClassifier(100, n_jobs=-1,
                                                        random_state=common.SEED + r))):
            p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            out[name].append(roc_auc_score(y, p))
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)),
                                                    float(np.percentile(v, 97.5))],
                "pass": bool(np.mean(v) <= CRITERION)} for k, v in out.items()}, n


def moments(Zs, Zt):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(Zs)), np.ones(len(Zt))]
    Z = np.vstack([Zs, Zt])
    aucs = []
    for j in range(Z.shape[1]):
        a = roc_auc_score(y, Z[:, j]) if Z[:, j].std() > 0 else 0.5
        aucs.append(max(a, 1 - a))
    j = int(np.argmax(aucs))
    return {"mean_gap_l2": float(np.linalg.norm(Zs.mean(0) - Zt.mean(0))),
            "cov_gap_fro": float(np.linalg.norm(np.cov(Zs.T) - np.cov(Zt.T))),
            "best_dim": j, "best_dim_auc": float(aucs[j]),
            "best_dim_range_overlap": overlap(Zs[:, j], Zt[:, j])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    cols = common.raw_columns()
    tr, _, _ = common.load_corpus_split()
    sg = pd.read_csv(os.path.join(common.MC_DIR, "data", "selfgen_dataset.csv"))
    sg = sg[sg["batch"] == WIN_BATCH]
    assert len(sg) == 23135, len(sg)
    import joblib
    pipe = joblib.load(common.model_paths(".r49v3_win")["bundle"])["pipeline"]

    Xs = {c: np.nan_to_num(pipe.transform(tr[tr["label_str"] == c][cols])) for c in CLASSES}
    Xt = {c: np.nan_to_num(pipe.transform(sg[sg["label_str"] == c][cols])) for c in CLASSES}

    table6, table7, n_used = {}, {}, {}
    for arm, suf in ARMS:
        f = (lambda X: X) if suf is None else (
            lambda X, _m=penultimate(suf): _m.predict(X.reshape(-1, 1, 41).astype("float32"),
                                                     batch_size=8192, verbose=0))
        table6[arm] = {}
        for c in CLASSES:
            Zs, Zt = f(Xs[c]), f(Xt[c])
            table6[arm][c], n_used[c] = domain_auc(Zs, Zt, a.reps)
            if c == "WebAttack" and suf is not None:
                table7[arm] = moments(Zs, Zt)
        print(f"  {arm}: " + " | ".join(
            f"{c} LR {table6[arm][c]['lr']['mean']:.4f} RF {table6[arm][c]['rf']['mean']:.4f}"
            for c in CLASSES), flush=True)

    # kok neden: girdi uzayinda ozellik basina AUC + aralik ortusmesi
    root = {}
    for c in CLASSES:
        rows = []
        raw_s, raw_t = tr[tr["label_str"] == c][cols], sg[sg["label_str"] == c][cols]
        y = np.r_[np.zeros(len(raw_s)), np.ones(len(raw_t))]
        for col in cols:
            v = np.r_[raw_s[col].values, raw_t[col].values]
            au = roc_auc_score(y, v) if v.std() > 0 else 0.5
            rows.append((col, max(au, 1 - au), overlap(raw_s[col].values, raw_t[col].values)))
        rows.sort(key=lambda r: (-r[1], r[2]))
        root[c] = [{"feature": r[0], "auc": round(r[1], 4), "range_overlap": round(r[2], 4)}
                   for r in rows[:5]]

    any_pass = any(v[c][m]["pass"] for v in table6.values() for c in CLASSES for m in ("lr", "rf")
                   if v is not table6["raw_input_41"])
    result = {
        "gate": "R50 Gate-0 conditional CORAL / CDAN-lite (no training)",
        "criterion": f"domain AUC <= {CRITERION} in penultimate Dense(32)",
        "domains": {"source": "CICIDS train split (batch_level_split)",
                    "target": f"selfgen {WIN_BATCH}", "balanced_n_per_domain": n_used,
                    "reps": a.reps},
        "table6_domain_auc": table6,
        "table7_webattack_moments": table7,
        "table7_memo_reference": MEMO_TABLE7,
        "root_cause_input_space_top5": root,
        "verdict": "PASS -> run GATE-1" if any_pass else "FAIL at Gate-0 -> GATE-1 not run",
        "provenance": common.provenance(
            [s for _, s in ARMS if s] + [".r49v3_win"],
            [common.DATA_CSV, os.path.join(common.MC_DIR, "data", "selfgen_dataset.csv")]),
    }
    common.write_json("r50_coral_gate0.json", result)
    print(result["verdict"])


if __name__ == "__main__":
    main()
