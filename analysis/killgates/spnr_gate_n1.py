# -*- coding: utf-8 -*-
"""
SPNR Gate-N(-1) -- Support-Preserving Nuisance Randomization on-kill-gate
(SIFIR EGITIM).

Soru: ESP32'nin Init_Win degerini korpusun KENDI WebAttack degerine sabitlemek,
uretim modelinin DOGRU SINIF (WebAttack / FTP-Patator) orani uzerinde etki
yaratiyor mu? Yaratmiyorsa SPNR tam pilotu (4 kol x 3 model retrain) KOSULMAZ.

On-kayitli kural: "T5 dogru-sinif orani hala .0000 ise pilotu KOSMA."

Kollar (yalniz saldirgan satirlari, SYN-gorulmus rejim):
  T1  mudahale yok
  T5f fwd := 29200          (SPNR'nin asil hedefi -- izole)
  T5b bwd := 28960
  T5  fwd := 29200, bwd := 28960   (korpus WebAttack'inin TEK gecerli cifti)

Ek olarak modelden bagimsiz sonsal: P(WebAttack | Init_Win_fwd = 29200, SYN).

Kullanim: python analysis/killgates/spnr_gate_n1.py
Cikti   : analysis/killgates/out/spnr_gate_n1.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

FWD, BWD = "Init_Win_bytes_forward", "Init_Win_bytes_backward"
ARMS = [  # arm adlari gate35 ile ortak -> onbellek paylasilir
    ("A0_production", "T1", []),
    ("T5f_fwd29200", "T5f", [(FWD, 29200)]),
    ("A1_bwd28960", "T5b", [(BWD, 28960)]),
    ("T5_fwd29200_bwd28960", "T5", [(FWD, 29200), (BWD, 28960)]),
]
EXPECTED = {"T1": (0.2035, 0.1172), "T5f": (0.2035, 0.1172),
            "T5b": (0.3625, 0.5934), "T5": (0.4092, 0.8828)}


def posterior():
    import pandas as pd
    df = pd.read_csv(common.DATA_CSV, usecols=["SYN Flag Count", FWD, BWD, "label_str"])
    s = df[df["SYN Flag Count"] > 0]
    wa = s[s["label_str"] == "WebAttack"]
    f = s[s[FWD] == 29200]
    b = s[s[BWD] == 28960]
    return {
        "webattack_syn_rows": int(len(wa)),
        "webattack_fwd_value_share": {str(int(k)): round(v, 4) for k, v in
                                      wa[FWD].value_counts(normalize=True).items()},
        "p_class_given_fwd29200": {k: round(v, 4) for k, v in
                                   f["label_str"].value_counts(normalize=True).items()},
        "n_fwd29200": int(len(f)),
        "p_webattack_given_bwd28960": round(float((b["label_str"] == "WebAttack").mean()), 4),
        "n_bwd28960": int(len(b)),
        "fwd_support_values_syn": int(s[FWD].nunique()),
        "fwd_rows_in_4000_8000": int(((s[FWD] >= 4000) & (s[FWD] <= 8000)
                                      & (s[FWD] != 8192)).sum()),
    }


def main():
    rw = {}
    for arm, tag, ov in ARMS:
        rw[tag] = common.run_realworld_arm(arm, "", ov)

    ccr = {tag: [s["correct_class_rate"]["rate"] for s in r["sets"]] for tag, r in rw.items()}
    rec = {tag: [round(s["binary_recall"]["rate"], 4) for s in r["sets"]]
           for tag, r in rw.items()}
    raw_hits = {tag: [sum(v.get(s["expected_class"], 0) for v in s["raw_argmax_attacker"].values())
                      for s in r["sets"]] for tag, r in rw.items()}
    pilot_killed = all(x == 0 for x in ccr["T5"])
    t5f_identical_to_t1 = all(
        a["binary_recall"] == b["binary_recall"] and a["alert_classes"] == b["alert_classes"]
        for a, b in zip(rw["T1"]["sets"], rw["T5f"]["sets"]))

    fid = [{"check": f"rw_recall[{t}]", "got": rec[t], "expected": list(e),
            "ok": all(abs(g - x) < 5e-5 for g, x in zip(rec[t], e))}
           for t, e in EXPECTED.items()]
    result = {
        "gate": "SPNR Gate-N(-1) (zero-training)",
        "preregistered_rule": "T5 correct-class rate still .0000 -> do NOT run SPNR pilot",
        "correct_class_rate": ccr,
        "binary_recall": rec,
        "raw_model_expected_class_argmax_hits": raw_hits,
        "T5f_packet_identical_to_T1": t5f_identical_to_t1,
        "verdict": "KILL (do not run SPNR pilot)" if pilot_killed else "PROCEED",
        "posterior_model_independent": posterior(),
        "realworld": rw,
        "fidelity": fid,
        "fidelity_all_ok": all(f["ok"] for f in fid),
        "provenance": common.provenance(("",), [common.DATA_CSV]),
    }
    common.write_json("spnr_gate_n1.json", result)
    print(f"ccr={ccr} recall={rec} T5f==T1:{t5f_identical_to_t1} -> {result['verdict']}")


if __name__ == "__main__":
    main()
