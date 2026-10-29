# -*- coding: utf-8 -*-
"""
R38 -- WebAttack recall'unun payload FAZINA gore ayristirilmasi (DG makalesi Tablo 4).
SIFIR EGITIM; uretim modelleri (MulticlassLSTM.keras / MulticlassRF.joblib).

Bulgu: build_multiclass_streaming_dataset.py TOP_K=6 batch'i HACIMDEN siraliyor;
Thursday/WebAttack icin secilen 6 batch'in (284/286/288/296/346/348) tamami
Brute-Force fazi. CICIDS'in kendi XSS (386-404) ve SQLi (414-418) fazlari diskte
duruyor ama egitime secilmemis. Bu betik ucunu ayni modelle, ayni ozellik
cikarimiyla puanlar.

Metrik: gt == WebAttack satirlarinda argmax == WebAttack (sinif recall'u),
Wilson %95 araligi ile.

  Brute-Force : korpus CSV'sindeki TUM WebAttack satirlari (egitim + test; bu
                faz egitimde VAR -- "in training set")
  XSS / SQLi  : ham batch pcap'lerinden, egitimdeki `_extract_multiclass` ile
                BIREBIR ayni cikarim ve ayni GT fonksiyonu (SCAN_DEFS).
                XSS fazi 10 batch (386-404) tasiyor; egitimdeki secim kuralinin
                aynisi (TOP_K=6, saldiri hacmine gore) uygulanir -> 386/390/394/
                396/400/404, n=3289. (Bu, 2026-08-14 olcumunun n'ini birebir
                veren TEK alt-kumedir; 10 batch'in tamami n=3960 verir ve
                `--xss-all` ile ayrica raporlanir.)
  ESP32       : registry seti esp32_webattack_20260814 (saldirgan satirlari).
                NOT: makaledeki n=16.322 satiri, artik diskte olmayan eski bir
                halka-tamponu yakalamasindan geliyordu; burada kalici registry
                seti (n=2.378) kullanilir.

Kullanim: python analysis/killgates/r38_phase_recall.py
Cikti   : analysis/killgates/out/r38_phase_recall.json
"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

XSS_BATCHES_ALL = [386, 388, 390, 392, 394, 396, 398, 400, 402, 404]
TOP_K = 6  # build_multiclass_streaming_dataset.TOP_K ile ayni secim kurali
SQLI_BATCHES = [414, 416, 418]
MEMO = {"brute_force": {"n": 6318, "lstm": 0.8593, "rf": 0.9462},
        "xss": {"n": 3289, "lstm": 0.7549, "rf": 0.3062},
        "sqli": {"n": 64, "lstm": 0.3281, "rf": 0.1719},
        "esp32": {"n": 16322, "lstm": 0.0}}


def extract_phase(batches):
    import build_multiclass_streaming_dataset as bm
    from _scan_train_candidates import SCAN_DEFS
    from validate_all_days import find_batch_day
    gt = {f"{d}/{f}": g for d, f, g, *_ in SCAN_DEFS}["Thursday/WebAttack"]
    rows = []
    for idx in batches:
        pcap = find_batch_day("Thursday", f"{idx:05d}")
        assert pcap, f"batch {idx} bulunamadi"
        with contextlib.redirect_stdout(io.StringIO()):
            rows += bm._extract_multiclass(pcap, gt, "WebAttack", "Thursday", idx)
    df = pd.DataFrame(rows)
    return df[df["label_str"] == "WebAttack"]


def extract_topk(batches, k):
    """Batch basina WebAttack satiri sayar, en hacimli k batch'i dondurur."""
    per = {b: extract_phase([b]) for b in batches}
    top = sorted(per, key=lambda b: -len(per[b]))[:k]
    return pd.concat([per[b] for b in sorted(top)]), sorted(top), {b: len(v) for b, v in per.items()}


def esp32_rows():
    """realworld kolundaki ile ayni: ana pas, saldirgan satirlari."""
    import pcap_validation_pilot as v1
    import validate_realworld_pcaps as vr
    with contextlib.redirect_stdout(io.StringIO()):
        import offline_test_multiclass as mc
    spec = [s for s in vr.REALWORLD_SETS if s["name"] == "esp32_webattack_20260814"][0]
    ft = mc.UniversalRealTimeFlowTableV5()
    rows = []
    for p in v1.read_pcap(spec["pcap"]):
        if mc._is_non_ip(p):
            continue
        mf, _ = ft.extract_universal_features(p)
        if mf is not None and p.get("ip.src") == spec["attacker_ip"]:
            rows.append({c: float(mf.get(c, 0.0)) for c in common.raw_columns()})
    return pd.DataFrame(rows)


def main():
    cols = common.raw_columns()
    models = common.load_models("")
    classes = models[4]
    wa = classes.index("WebAttack")

    corpus = pd.read_csv(common.DATA_CSV)
    xss_df, xss_top, xss_per_batch = extract_topk(XSS_BATCHES_ALL, TOP_K)
    phases = {"brute_force": corpus[corpus["label_str"] == "WebAttack"],
              "xss": None,
              "sqli": extract_phase(SQLI_BATCHES),
              "esp32": esp32_rows()}
    phases["xss"] = xss_df
    phases["xss_all_10_batches"] = extract_phase(XSS_BATCHES_ALL)
    out = {}
    for name, df in phases.items():
        P, _ = common.predict_all(models, df[cols])
        out[name] = {"n": int(len(df)),
                     "lstm": common.rate(int((P["lstm"].argmax(1) == wa).sum()), len(df)),
                     "rf": common.rate(int((P["rf"].argmax(1) == wa).sum()), len(df)),
                     "fwd_pkt_len_max_unique": int(df["Fwd Packet Length Max"].nunique())}
        print(f"  {name}: n={len(df)} LSTM={out[name]['lstm']['rate']:.4f} "
              f"RF={out[name]['rf']['rate']:.4f}", flush=True)

    fid = []
    for k in ("brute_force", "xss", "sqli"):
        ok = (out[k]["n"] == MEMO[k]["n"]
              and abs(out[k]["lstm"]["rate"] - MEMO[k]["lstm"]) < 5e-5
              and abs(out[k]["rf"]["rate"] - MEMO[k]["rf"]) < 5e-5)
        fid.append({"check": k, "ok": ok, "got": [out[k]["n"], round(out[k]["lstm"]["rate"], 4),
                                                   round(out[k]["rf"]["rate"], 4)],
                    "expected": [MEMO[k]["n"], MEMO[k]["lstm"], MEMO[k]["rf"]]})
    result = {
        "gate": "R38 WebAttack recall by payload phase (zero-training)",
        "paper_table": "dataset_generalization Table 4",
        "metric": "class recall: argmax == WebAttack on gt == WebAttack rows",
        "phases": out,
        "xss_selection": {"rule": f"top-{TOP_K} batches by attack volume (same as training)",
                          "selected": xss_top, "per_batch_rows": xss_per_batch},
        "esp32_note": "registry set (n=2378) replaces the paper's n=16322 ring-buffer "
                      "capture, which is no longer on disk",
        "monotone_collapse": out["brute_force"]["rf"]["rate"] > out["xss"]["rf"]["rate"]
                             > out["sqli"]["rf"]["rate"],
        "fidelity": fid,
        "fidelity_all_ok": all(f["ok"] for f in fid),
        "provenance": common.provenance(("",), [common.DATA_CSV]),
    }
    common.write_json("r38_phase_recall.json", result)


if __name__ == "__main__":
    main()
