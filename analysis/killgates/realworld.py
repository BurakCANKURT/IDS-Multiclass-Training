# -*- coding: utf-8 -*-
"""
killgates/realworld.py
======================
Tek bir gercek-dunya KOLUNU kosar (alt-surec; common.run_realworld_arm cagirir).

Metrik hesabi `analysis/validate_realworld_pcaps.py` ile BIREBIR aynidir (registry,
packet_key, predicted_class ve "src_ip == attacker_ip" GT kurali oradan import
edilir). Tek ek: istege bagli OZELLIK MUDAHALESI.

Mudahale (--override FEATURE=VALUE, tekrarlanabilir)
----------------------------------------------------
`UniversalRealTimeFlowTableV5.extract_universal_features` sinif duzeyinde
sarmalanir -> hem LSTM on-hesap pasi hem ana pas kapsanir. Deger yalnizca:
  * paketin src_ip'si setin attacker_ip'si ise, VE
  * akis SYN-gorulmus rejimdeyse (akista >=1 SYN sayilmis)
degistirilir. Arka plan satirlarina dokunulmaz -> arka plan FAR'i her kolda
sabit kalmalidir (kapsam kaniti; cikti JSON'unda raporlanir).

Model kolu MC_EVAL_SUFFIX ortam degiskeniyle secilir (offline_test_multiclass
bunu import aninda okur).
"""

import argparse
import contextlib
import io
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import common  # noqa: E402  (sys.path'i ayarlar)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pcap_validation_pilot as v1  # noqa: E402
import validate_realworld_pcaps as vr  # noqa: E402
import offline_test_multiclass as mc  # noqa: E402


def install_override(overrides, attacker_ip, recorder):
    """Sinif metodunu sarar. recorder[id(table)] -> saldirgan satirlarinin
    (mudahale SONRASI) ozellik sozlukleri; ham model argmax'i icin."""
    cls = mc.UniversalRealTimeFlowTableV5
    orig = cls.extract_universal_features
    stats = {"attacker_rows": 0, "syn_regime_rows": 0, "overridden_rows": 0}

    def patched(self, packet):
        mf, fk = orig(self, packet)
        if mf is None or packet.get("ip.src") != attacker_ip:
            return mf, fk
        stats["attacker_rows"] += 1
        st = getattr(self, "_flag_counts", {}).get(fk)
        if st is not None and st.get("syn", 0) > 0:
            stats["syn_regime_rows"] += 1
            if overrides:
                stats["overridden_rows"] += 1
                for feat, val in overrides:
                    mf[feat] = float(val)
        recorder.setdefault(id(self), []).append(
            [float(mf.get(c, 0.0)) for c in mc.RAW_COLUMNS_V5])
        return mf, fk

    cls.extract_universal_features = patched
    return stats


def raw_argmax(records, rf, lstm, pipeline, classes):
    """Kapisiz ham model argmax dagilimi (saldirgan satirlari)."""
    if not records:
        return {}
    X = pd.DataFrame(records, columns=mc.RAW_COLUMNS_V5)
    Xs = np.nan_to_num(pipeline.transform(X))
    out = {"rf": Counter(np.asarray(classes)[rf.predict_proba(Xs).argmax(1)])}
    hgb_path = common.model_paths(os.environ.get("MC_EVAL_SUFFIX", ""))["hgb"]
    if os.path.exists(hgb_path):
        import joblib
        out["hgb"] = Counter(np.asarray(classes)[joblib.load(hgb_path).predict_proba(Xs).argmax(1)])
    lp = lstm.predict(Xs.reshape(-1, 1, Xs.shape[1]).astype("float32"), batch_size=4096, verbose=0)
    out["lstm"] = Counter(np.asarray(classes)[lp.argmax(1)])
    return {k: dict(v) for k, v in out.items()}


def flow_id(p):
    """Yonsuz 5-li (run_offline'in akis anahtariyla ayni kural)."""
    s, d = p.get("ip.src") or p.get("ipv6.src", ""), p.get("ip.dst") or p.get("ipv6.dst", "")
    sp = p.get("tcp.srcport") or p.get("udp.srcport") or "0"
    dp = p.get("tcp.dstport") or p.get("udp.dstport") or "0"
    return "|".join(sorted([f"{s}:{sp}", f"{d}:{dp}"]) + [str(p.get("ip.proto", ""))])


def score_set(spec, overrides, models, dump_path=None):
    rf, lstm, pipeline, classes, n_classes, benign_idx = models
    recorder = {}
    orig = mc.UniversalRealTimeFlowTableV5.extract_universal_features
    stats = install_override(overrides, spec["attacker_ip"], recorder)
    try:
        packets = v1.read_pcap(spec["pcap"])
        pkt_by_key = {}
        for i, p in enumerate(packets):
            pkt_by_key.setdefault(vr.packet_key(p, i), p)
        with contextlib.redirect_stdout(io.StringIO()):
            results = mc.run_offline(rf, lstm, pipeline, classes, n_classes,
                                     benign_idx, packets)
    finally:
        mc.UniversalRealTimeFlowTableV5.extract_universal_features = orig

    atk = Counter()
    bg = Counter()
    pred_counter = Counter()
    risks = []
    dump = []
    for r in results:
        src_pkt = pkt_by_key.get(r["packet_no"])
        if src_pkt is None:
            continue
        alert = bool(r["is_alert"] and not r["is_ood"])
        if dump_path:
            dump.append((r["packet_no"], v1._to_float(src_pkt.get("frame.time_epoch"), 0.0),
                         flow_id(src_pkt), r["src_ip"] == spec["attacker_ip"], alert,
                         bool(r["is_ood"]), bool(r["is_volumetric"])))
        if r["src_ip"] == spec["attacker_ip"]:
            atk["n"] += 1
            risks.append(float(r["risk_score"]))
            atk["ood"] += int(bool(r["is_ood"]))
            if alert:
                atk["alert"] += 1
                atk["vol" if r["is_volumetric"] else "behav"] += 1
                pc = vr.predicted_class(r.get("prediction"), classes)
                pred_counter[str(pc)] += 1
                if pc == spec["expected_class"]:
                    atk["correct"] += 1
        else:
            bg["n"] += 1
            bg["alert"] += int(alert)

    if dump_path:
        pd.DataFrame(dump, columns=["packet_no", "t", "flow", "is_attacker", "alert",
                                    "is_ood", "is_volumetric"]).to_csv(
            dump_path, index=False, compression="gzip")

    # ana pasin tablosu en son olusturulan ornektir
    main_records = recorder[list(recorder)[-1]] if recorder else []
    return {
        "name": spec["name"],
        "pcap_sha256": common.sha256(spec["pcap"]),
        "expected_class": spec["expected_class"],
        "attacker_packets": atk["n"],
        "binary_recall": common.rate(atk["alert"], atk["n"]),
        "behavioral_alerts": atk["behav"],
        "volumetric_alerts": atk["vol"],
        "correct_class_rate": common.rate(atk["correct"], atk["n"]),
        "attacker_ood": atk["ood"],
        "attacker_risk_p50": float(np.median(risks)) if risks else None,
        "alert_classes": dict(pred_counter),
        "background_far": common.rate(bg["alert"], bg["n"]),
        "override_stats": stats,
        "raw_argmax_attacker": raw_argmax(main_records, rf, lstm, pipeline, classes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--sets", default="")
    ap.add_argument("--dump", action="store_true",
                    help="paket-duzeyi karar dokumu: out/packets/<arm>__<set>.csv.gz")
    a = ap.parse_args()

    overrides = []
    for o in a.override:
        k, v = o.rsplit("=", 1)
        assert k in mc.RAW_COLUMNS_V5, f"bilinmeyen ozellik: {k}"
        overrides.append((k, float(v)))

    sets = vr.REALWORLD_SETS
    if a.sets:
        wanted = set(a.sets.split(","))
        sets = [s for s in sets if s["name"] in wanted]

    with contextlib.redirect_stdout(io.StringIO()):
        models = mc.load_multiclass_models()
    models = (models[0], models[1], models[2], list(models[3]), models[4], models[5])

    suffix = os.environ.get("MC_EVAL_SUFFIX", "")
    out = {"arm": a.arm, "suffix": suffix, "overrides": overrides, "sets": [],
           "provenance": common.provenance((suffix,))}
    pdir = os.path.join(common.OUT_DIR, "packets")
    os.makedirs(pdir, exist_ok=True)
    for spec in sets:
        if not os.path.exists(spec["pcap"]):
            raise FileNotFoundError(spec["pcap"])
        dp = os.path.join(pdir, f"{a.arm}__{spec['name']}.csv.gz") if a.dump else None
        out["sets"].append(score_set(spec, overrides, models, dp))
        s = out["sets"][-1]
        print(f"  {s['name']}: recall={s['binary_recall']['rate']:.4f} "
              f"bgFAR={s['background_far']['rate']:.5f}", flush=True)
    common.write_json(a.out, out)


if __name__ == "__main__":
    main()
