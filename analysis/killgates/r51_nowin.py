# -*- coding: utf-8 -*-
"""
R51 -- `.r51_nowin` retrain pilotu (DG makalesi Tablo 5).

Hipotez (GATE-35'ten): Init_Win_bytes_forward/backward bir "kurban parmak izi"
kisayolu; RF+HGB bu iki ozellik notrlenerek yeniden egitilirse gercek-dunya
ikili recall'i artar.

On-kayitli kapilar:
  GATE-A (korpus-ici bedel)  : RF ikili recall >= 0.85
  GATE-B (gercek-dunya kazanc): esp32_webattack VE esp32_ftp ikili recall > 0.25
  GATE-C (bedel tavani)      : arka plan FAR artisi <= 2x   (B kalirsa kosulmaz)

Mudahale: Init_Win sutunlari (idx 29, 30) OLCEKLENMIS uzayda 0.0'a sabitlenir,
sutun sayisi 41 kalir (ONNX/konsey/OOD/LSTM yuzeyi degismez). LSTM, scaler ve
OOD referansi uretimden bit-birebir kopyadir -- yalniz RF+HGB yeniden egitilir.

Kullanim:
    python analysis/killgates/r51_nowin.py            # diskteki .r51_nowin modelleriyle
    python analysis/killgates/r51_nowin.py --retrain  # ayrica RF+HGB'yi sifirdan egitip
                                                      # diskteki adayla ozdesligini dogrula
Cikti: analysis/killgates/out/r51_nowin.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SUFFIX = ".r51_nowin"
INIT_WIN = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

# Verdict memosundaki (2026-08-21) rakamlar -- sadakat kontrolu
EXPECTED = {
    "corpus_rf_recall": (0.9324, 0.9246),
    "corpus_rf_fpr": (0.00417, 0.00760),
    "rw_recall": {"esp32_webattack_20260814": (0.2035, 0.2035),
                  "esp32_ftp_patator_20260814": (0.1172, 0.1136)},
    "rw_bg_far": {"esp32_webattack_20260814": (0.00635, 0.01531),
                  "esp32_ftp_patator_20260814": (0.00252, 0.00340)},
}


def retrain_and_compare(tr, te, classes, cols):
    """RF+HGB'yi uretim hiperparametreleriyle yeniden egitir (bellekte) ve
    diskteki .r51_nowin adayiyla test-bolmesi argmax ozdesligini olcer."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    rf_d, hgb_d, _, pipeline, _ = common.load_models(SUFFIX)
    idx = [cols.index(c) for c in INIT_WIN]
    Xtr = pipeline.transform(tr[cols]); Xtr[:, idx] = 0.0
    Xte = pipeline.transform(te[cols]); Xte[:, idx] = 0.0
    ytr = tr["label"].values.astype(int)
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=10, max_depth=25,
                                max_features=0.5, random_state=42,
                                class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=13,
                                         class_weight="balanced").fit(Xtr, ytr)
    return {
        "rf_argmax_agreement": float((rf.predict(Xte) == rf_d.predict(Xte)).mean()),
        "hgb_argmax_agreement": float((hgb.predict(Xte) == hgb_d.predict(Xte)).mean()),
        "rf_initwin_importance": [float(rf.feature_importances_[i]) for i in idx],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()

    tr, te, classes = common.load_corpus_split()
    cols = common.raw_columns()
    y = te["label_str"].values

    corpus = {}
    for arm, suf in (("production", ""), ("r51_nowin", SUFFIX)):
        P, _ = common.predict_all(common.load_models(suf), te[cols])
        corpus[arm] = {m: common.binary_metrics(P[m], y, classes) for m in ("rf", "hgb")}

    rw = {"production": common.run_realworld_arm("A0_production", ""),
          "r51_nowin": common.run_realworld_arm("r51_nowin", SUFFIX)}

    # --- Kapilar ---
    rf_rec = corpus["r51_nowin"]["rf"]["binary_recall"]["rate"]
    rw_c = {s["name"]: s for s in rw["r51_nowin"]["sets"]}
    rw_b = {s["name"]: s for s in rw["production"]["sets"]}
    gate_a = rf_rec >= 0.85
    gate_b = all(s["binary_recall"]["rate"] > 0.25 for s in rw_c.values())
    far_ratio = {n: rw_c[n]["background_far"]["rate"] / rw_b[n]["background_far"]["rate"]
                 for n in rw_c}
    gate_c = all(r <= 2.0 for r in far_ratio.values())

    # --- Sadakat ---
    fid = []
    def chk(name, got, exp, tol=5e-5):
        ok = abs(got - exp) <= tol
        fid.append({"check": name, "got": round(got, 5), "expected": exp, "ok": ok})
    for i, arm in enumerate(("production", "r51_nowin")):
        chk(f"corpus_rf_recall[{arm}]", corpus[arm]["rf"]["binary_recall"]["rate"],
            EXPECTED["corpus_rf_recall"][i])
        chk(f"corpus_rf_fpr[{arm}]", corpus[arm]["rf"]["fpr"]["rate"],
            EXPECTED["corpus_rf_fpr"][i])
        sets = {s["name"]: s for s in rw[arm]["sets"]}
        for n, e in EXPECTED["rw_recall"].items():
            chk(f"rw_recall[{arm}][{n}]", sets[n]["binary_recall"]["rate"], e[i])
        for n, e in EXPECTED["rw_bg_far"].items():
            chk(f"rw_bg_far[{arm}][{n}]", sets[n]["background_far"]["rate"], e[i])

    result = {
        "gate": "R51 .r51_nowin retrain pilot",
        "paper_table": "dataset_generalization Table 5",
        "corpus_test_split": {"n": int(len(te)), "attack": int((y != "BENIGN").sum()),
                              "benign": int((y == "BENIGN").sum())},
        "corpus": corpus,
        "realworld": rw,
        "background_far_ratio": far_ratio,
        "gates": {"GATE-A (RF recall >= .85)": gate_a,
                  "GATE-B (both ESP32 recall > .25)": gate_b,
                  "GATE-C (bg FAR ratio <= 2x)": gate_c},
        "verdict": "REJECT" if not (gate_a and gate_b and gate_c) else "ACCEPT",
        "fidelity": fid,
        "fidelity_all_ok": all(f["ok"] for f in fid),
        "provenance": common.provenance(("", SUFFIX), [common.DATA_CSV]),
    }
    if a.retrain:
        result["retrain_check"] = retrain_and_compare(tr, te, classes, cols)

    common.write_json("r51_nowin.json", result)
    print(f"GATE-A={gate_a} GATE-B={gate_b} GATE-C={gate_c} -> {result['verdict']} "
          f"| sadakat={'OK' if result['fidelity_all_ok'] else 'SAPMA'}")


if __name__ == "__main__":
    main()
