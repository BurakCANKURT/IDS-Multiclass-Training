# -*- coding: utf-8 -*-
"""
GATE-35 -- Init_Win_bytes_backward "benign kisayolu" (SIFIR EGITIM).

Soru: uretim modeli `Init_Win_bytes_backward`'i saldiri ozelligi olarak mi,
yoksa CICIDS kurban-VM SYN-ACK pencere parmak izi olarak mi kullaniyor?

Uc olcum:
  S  (destek)   : SYN rejiminde (SYN Flag Count > 0) bwd degerlerinin P(saldiri)
                  dagilimi -- kac deger saldiri-kapasiteli, kaci saf BENIGN.
  A0-A6 (gercek): ESP32 saldirgan satirlarinda bwd := deger; ikili recall.
                  A5/A6 = saf-benign kontrol degerleri. A6 esp32_ftp'nin kendi
                  yerel degeri -> A0 ile BIT-BIREBIR ayni olmali (no-op kontrolu).
  C0-C3 (korpus): uretim modelleri, test bolmesinde bwd bozularak
                  (native / permute / :=65535 / :=28960).

Bu gate bagimlilik (reliance) olcer, zorunluluk (necessity) DEGIL -- zorunluluk
icin bkz. r51_nowin.py (36b. KURAL).

Kullanim: python analysis/killgates/gate35_initwin_bwd.py
Cikti   : analysis/killgates/out/gate35_initwin_bwd.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

BWD = "Init_Win_bytes_backward"
ARMS = [  # (kol, enjekte edilen deger veya None)
    ("A0_production", None),
    ("A1_bwd28960", 28960),
    ("A2_bwd8192", 8192),
    ("A3_bwd0", 0),
    ("A4_bwd29200", 29200),
    ("A5_bwd42780_ctrl", 42780),
    ("A6_bwd65535_ctrl", 65535),
]
EXPECTED_RW = {  # memo 2026-08-21, ikili recall (WA, FTP)
    "A0_production": (0.2035, 0.1172), "A1_bwd28960": (0.3625, 0.5934),
    "A2_bwd8192": (0.3696, 0.3297), "A3_bwd0": (0.3608, 0.4359),
    "A4_bwd29200": (0.3612, 0.6007), "A5_bwd42780_ctrl": (0.3608, 0.2308),
    "A6_bwd65535_ctrl": (0.2035, 0.1172),
}
EXPECTED_CORPUS_RF = {"C0_native": 0.9324, "C1_permute": 0.7903,
                      "C2_bwd65535": 0.4113, "C3_bwd28960": 0.9176}


def support_table():
    import pandas as pd
    df = pd.read_csv(common.DATA_CSV, usecols=["SYN Flag Count", BWD, "label_str"])
    s = df[df["SYN Flag Count"] > 0]
    atk = s["label_str"] != "BENIGN"
    g = pd.DataFrame({"n": s.groupby(BWD).size(),
                      "p_attack": atk.groupby(s[BWD]).mean()})
    capable = g[g["p_attack"] > 0].sort_values("n", ascending=False)
    pure = g[g["p_attack"] == 0].sort_values("n", ascending=False)
    top4 = capable.head(4)
    return {
        "syn_regime_rows": int(len(s)), "syn_regime_attack_rows": int(atk.sum()),
        "n_values": int(len(g)),
        "attack_capable_values": int(len(capable)),
        "pure_benign_values": int(len(pure)),
        "attack_rows_in_top4_values": int(s[atk][BWD].isin(top4.index).sum()),
        "top4_attack_values": {str(int(k)): {"n": int(v.n), "p_attack": round(v.p_attack, 4)}
                               for k, v in top4.iterrows()},
        "largest_pure_benign_values": {str(int(k)): int(v.n)
                                       for k, v in pure.head(6).iterrows()},
        "p_65535_given_benign": round(float((s[~atk][BWD] == 65535).mean()), 4),
    }


def corpus_arms():
    import numpy as np
    tr, te, classes = common.load_corpus_split()
    cols = common.raw_columns()
    y = te["label_str"].values
    models = common.load_models("")
    rng = np.random.RandomState(common.SEED)
    arms = {"C0_native": None, "C1_permute": "perm",
            "C2_bwd65535": 65535.0, "C3_bwd28960": 28960.0}
    out = {}
    for name, v in arms.items():
        X = te[cols].copy()
        if v == "perm":
            X[BWD] = rng.permutation(X[BWD].values)
        elif v is not None:
            X[BWD] = v
        P, _ = common.predict_all(models, X)
        out[name] = {m: common.binary_metrics(P[m], y, classes) for m in ("rf", "lstm")}
    return out


def main():
    support = support_table()
    rw = {}
    for arm, v in ARMS:
        ov = [] if v is None else [(BWD, v)]
        rw[arm] = common.run_realworld_arm(arm, "", ov)
    corpus = corpus_arms()

    fid = []
    for arm, (wa, ftp) in EXPECTED_RW.items():
        got = [s["binary_recall"]["rate"] for s in rw[arm]["sets"]]
        fid.append({"check": f"rw_recall[{arm}]", "got": [round(g, 4) for g in got],
                    "expected": [wa, ftp],
                    "ok": abs(got[0] - wa) < 5e-5 and abs(got[1] - ftp) < 5e-5})
    for arm, e in EXPECTED_CORPUS_RF.items():
        g = corpus[arm]["rf"]["binary_recall"]["rate"]
        # C1 rastgele permutasyondur: orijinal tohum belgelenmemis -> 0.01 tolerans
        tol = 0.01 if arm == "C1_permute" else 5e-5
        fid.append({"check": f"corpus_rf_recall[{arm}]", "got": round(g, 4),
                    "expected": e, "ok": abs(g - e) < tol})
    # kapsam kaniti: arka plan FAR tum kollarda sabit
    bg = {arm: [round(s["background_far"]["rate"], 6) for s in r["sets"]]
          for arm, r in rw.items()}
    scope_ok = len({tuple(v) for v in bg.values()}) == 1
    # no-op kontrolu: A6 esp32_ftp == A0 esp32_ftp
    a0f, a6f = rw["A0_production"]["sets"][1], rw["A6_bwd65535_ctrl"]["sets"][1]
    noop_ok = (a0f["binary_recall"] == a6f["binary_recall"]
               and a0f["raw_argmax_attacker"] == a6f["raw_argmax_attacker"])

    summary = {arm: {s["name"]: round(s["binary_recall"]["rate"], 4) for s in r["sets"]}
               for arm, r in rw.items()}
    result = {
        "gate": "GATE-35 Init_Win_bwd victim fingerprint (zero-training)",
        "support_syn_regime": support,
        "realworld_recall_summary": summary,
        "realworld": rw,
        "corpus": corpus,
        "scope_check_background_far_constant": scope_ok,
        "background_far_by_arm": bg,
        "noop_check_A6_equals_A0_on_ftp": noop_ok,
        "fidelity": fid,
        "fidelity_all_ok": all(f["ok"] for f in fid) and scope_ok and noop_ok,
        "provenance": common.provenance(("",), [common.DATA_CSV]),
    }
    common.write_json("gate35_initwin_bwd.json", result)
    print("recall:", summary)
    print(f"kapsam={scope_ok} no-op={noop_ok} sadakat={result['fidelity_all_ok']}")


if __name__ == "__main__":
    main()
