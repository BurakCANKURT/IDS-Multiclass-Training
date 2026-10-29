# -*- coding: utf-8 -*-
"""
GATE-AT-0 -- aktif akis zaman asimi (active flow timeout) kill-gate'i
(DG makalesi Tablo 1). SIFIR EGITIM; uretim kodu DEGISMEZ.

Prototip: UniversalRealTimeFlowTableV5'in alt sinifi; akis `start_time`'dan T
saniye sonra zorla silinir (bayrak sayaclari dahil), sonraki paket yeni akis
acar (CICFlowMeter active-timeout semantigi). `run_offline` icindeki tablo
sinifi bu alt sinifla degistirilir -> dispatcher/kapilar/OOD degismeden kosar.

On-kayitli kriterler
  K1  CICIDS satir degisimi  < %5      (Wednesday 00300 CLEAN + 00800 ATTACK)
  K2  canli uzun-omurlu akis OOD p95 < 21   (kullanicinin 2 saatlik yakalamasi)
  K3  Slowloris/Slowhttptest DR bozulmasi > -2 puan  (held-out batch'ler)
  Planlanmamis yan etki: canli nihai alarm orani ve kaynagi (olgun/genc akis)

K3 davranissal-dal proxy'si: LSTM argmax != BENIGN VE konsey vetosu yok
(RF & HGB P(benign) >= .95 degil) VE OOD <= 30; maske: volumetrik-degil ve
act_data_pkt_fwd >= 1.

Kullanim: python analysis/killgates/gate_at0_active_timeout.py [--quick]
Cikti   : analysis/killgates/out/gate_at0_active_timeout.json
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pcap_validation_pilot as v1  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import offline_test_multiclass as mc  # noqa: E402

BASE = mc.UniversalRealTimeFlowTableV5
T_GRID = [None, 300, 120, 30, 10, 3]           # None = uretim (sonsuz)
K1_BATCHES = ["00300", "00800"]
HELDOUT = ["00595", "00600", "00610", "00630", "00635", "00640"]
LIVE_PCAP = os.path.join(common.PROJECT_DIR, "data", "realworld",
                         "capture_r26_phase2_softvote_20260814_1328.pcapng")
MATURE_ACT = 143  # egitim BENIGN act_data_pkt_fwd p99.9
MEMO = {"live_inf": {"ood_p50": 15.280, "ood_p95": 204.149, "proxy_alarms": 60,
                     "proxy_alarm_rate": 0.00086, "splits": 0},
        "live_T3": {"ood_p95": 115.541, "proxy_alarms": 1018, "proxy_alarm_rate": 0.01456,
                    "splits": 6862},
        "k3_inf": {"dr": 0.8193, "tp": 2925, "fn": 645},
        "k3_T3": {"dr": 0.9209, "tp": 2433}}


def make_table_class(T):
    class TimeoutTable(BASE):
        instances = []

        def __init__(self):
            super().__init__()
            self.n_splits = 0
            self.rec = {}
            TimeoutTable.instances.append(self)

        def extract_universal_features(self, packet):
            if T is not None:
                s, d = packet.get("ip.src", "0.0.0.0"), packet.get("ip.dst", "0.0.0.0")
                sp = v1._to_int(packet.get("tcp.srcport") or packet.get("udp.srcport"), 0)
                dp = v1._to_int(packet.get("tcp.dstport") or packet.get("udp.dstport"), 0)
                key = tuple(sorted([f"{s}:{sp}", f"{d}:{dp}"])) + (
                    v1._to_int(packet.get("ip.proto"), 6),)
                f = self.flows.get(key)
                now = v1._to_float(packet.get("frame.time_epoch"), 0.0) * 1_000_000
                if f is not None and now - f["start_time"] > T * 1_000_000:
                    del self.flows[key]
                    self._flag_counts.pop(key, None)
                    self.n_splits += 1
            mf, fk = super().extract_universal_features(packet)
            if mf is not None:
                self.rec[packet.get("frame.number")] = [float(mf.get(c, 0.0))
                                                        for c in mc.RAW_COLUMNS_V5]
            return mf, fk
    return TimeoutTable


def run(packets, T, models):
    cls = make_table_class(T)
    mc.UniversalRealTimeFlowTableV5 = cls
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = mc.run_offline(*models, packets)
    finally:
        mc.UniversalRealTimeFlowTableV5 = BASE
    main = cls.instances[-1]
    df = pd.DataFrame(res)
    feats = pd.DataFrame.from_dict(main.rec, orient="index", columns=mc.RAW_COLUMNS_V5)
    feats.index = feats.index.astype(str)
    df["_k"] = df["packet_no"].astype(str)
    df = df.join(feats[["act_data_pkt_fwd"]], on="_k")
    return df, feats, main.n_splits


def k1_row_change(T_values):
    out = {"batch_duration_s": {}, "arms": {}}
    from build_streaming_dataset import TRAIN_BATCHES
    for idx in TRAIN_BATCHES[:12]:
        p = v1.read_pcap(v1.find_batch(idx))
        t = [v1._to_float(x.get("frame.time_epoch"), 0.0) for x in p]
        out["batch_duration_s"][idx] = max(t) - min(t)
    d = np.array(list(out["batch_duration_s"].values()))
    out["batch_duration_summary"] = {"p50": float(np.median(d)), "min": float(d.min()),
                                     "max": float(d.max())}
    for T in T_values:
        changed = total = splits = 0
        for idx in K1_BATCHES:
            pk = v1.read_pcap(v1.find_batch(idx))
            ref_t, t_t = BASE(), make_table_class(T)()
            for p in pk:
                a, _ = ref_t.extract_universal_features(p)
                b, _ = t_t.extract_universal_features(p)
                if a is None:
                    continue
                total += 1
                changed += int(any(a[c] != b[c] for c in mc.RAW_COLUMNS_V5))
            splits += t_t.n_splits
        out["arms"][str(T)] = {"splits": splits, "rows": total,
                               "changed_frac": changed / total if total else 0.0}
    return out


def behavioral_proxy(df, feats, models, hgb):
    """K3 ile ayni davranissal-dal proxy'si: LSTM argmax != BENIGN, konsey
    vetosu yok, OOD <= 30. (2026-09-01 memosunun 'nihai alarm'i BU idi.)"""
    rf, lstm, pipeline, _, _, benign_idx = models
    X = np.nan_to_num(pipeline.transform(feats.loc[df["_k"]]))
    pl = lstm.predict(X.reshape(-1, 1, 41).astype("float32"), batch_size=8192, verbose=0)
    veto = ((rf.predict_proba(X)[:, benign_idx] >= .95)
            & (hgb.predict_proba(X)[:, benign_idx] >= .95))
    ood_ok = df["ood_distance"].fillna(0).astype(float).values <= 30
    return (pl.argmax(1) != benign_idx) & ~veto & ood_ok


def k2_live(T_values, models):
    import joblib
    hgb = joblib.load(common.model_paths("")["hgb"])
    pk = v1.read_pcap(LIVE_PCAP)
    out = {"packets": len(pk)}
    for T in T_values:
        df, feats, splits = run(pk, T, models)
        ood = df["ood_distance"].dropna().astype(float)
        alarm = df["is_alert"].astype(bool) & ~df["is_ood"].astype(bool)
        proxy = pd.Series(behavioral_proxy(df, feats, models, hgb), index=df.index)
        mature = df["act_data_pkt_fwd"] > MATURE_ACT
        young = df["act_data_pkt_fwd"] <= 2
        out[str(T)] = {
            "splits": splits,
            "ood_p50": float(ood.median()), "ood_p95": float(ood.quantile(.95)),
            "ood_band_5_21": float(((ood >= 5) & (ood <= 21)).mean()),
            "ood_gt30": float((ood > 30).mean()),
            "alarm_rate": float(alarm.mean()), "alarms": int(alarm.sum()),
            "mature_frac": float(mature.mean()),
            "mature_ood_p50": float(df.loc[mature, "ood_distance"].dropna().median())
            if mature.any() else None,
            "mature_alarm_rate": float(alarm[mature].mean()) if mature.any() else 0.0,
            "young_alarm_rate": float(alarm[young].mean()),
            "alarms_in_young_bucket": int((alarm & young).sum()),
            # memo (2026-09-01) bu proxy'yi 'nihai alarm' olarak raporladi
            "proxy_alarms": int(proxy.sum()),
            "proxy_alarm_rate_per_packet": float(proxy.sum() / len(pk)),
            "proxy_alarms_in_young_bucket": int((proxy & young).sum()),
        }
        print(f"  K2 T={T}: p95={out[str(T)]['ood_p95']:.1f} alarm={out[str(T)]['alarm_rate']:.5f}",
              flush=True)
    return out


def k3_slow_dos(T_values, models):
    rf, lstm, pipeline, classes, _, benign_idx = models
    import joblib
    hgb = joblib.load(common.model_paths("")["hgb"])
    out = {}
    for T in T_values:
        agg = dict(tp=0, fn=0, fp=0, tn=0)
        splits = 0
        for idx in HELDOUT:
            pk = v1.read_pcap(v1.find_batch(idx))
            gt = {str(p.get("frame.number")): v1.gt_label(p) for p in pk}
            df, feats, s = run(pk, T, models)
            splits += s
            df = df[~df["is_volumetric"].astype(bool) & (df["act_data_pkt_fwd"] >= 1)]
            X = np.nan_to_num(pipeline.transform(feats.loc[df["_k"]]))
            pl = lstm.predict(X.reshape(-1, 1, 41).astype("float32"), batch_size=8192, verbose=0)
            veto = ((rf.predict_proba(X)[:, benign_idx] >= .95)
                    & (hgb.predict_proba(X)[:, benign_idx] >= .95))
            ood_ok = df["ood_distance"].fillna(0).astype(float).values <= 30
            alarm = (pl.argmax(1) != benign_idx) & ~veto & ood_ok
            y = df["_k"].map(gt).values.astype(int) == 1
            agg["tp"] += int((alarm & y).sum()); agg["fn"] += int((~alarm & y).sum())
            agg["fp"] += int((alarm & ~y).sum()); agg["tn"] += int((~alarm & ~y).sum())
        out[str(T)] = {**agg, "splits": splits,
                       "dr": agg["tp"] / max(agg["tp"] + agg["fn"], 1),
                       "far": agg["fp"] / max(agg["fp"] + agg["tn"], 1)}
        print(f"  K3 T={T}: DR={out[str(T)]['dr']:.4f} TP={agg['tp']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="yalniz T in {inf, 3}")
    a = ap.parse_args()
    Ts = [None, 3] if a.quick else T_GRID
    with contextlib.redirect_stdout(io.StringIO()):
        m = mc.load_multiclass_models()
    models = (m[0], m[1], m[2], list(m[3]), m[4], m[5])

    k1 = k1_row_change(Ts[1:])
    k2 = k2_live(Ts, models)
    k3 = k3_slow_dos(Ts, models)

    k1_pass = all(v["changed_frac"] < 0.05 for v in k1["arms"].values())
    k2_pass = any(k2[str(T)]["ood_p95"] < 21 for T in Ts[1:])
    base_dr = k3["None"]["dr"]
    k3_pass = all(k3[str(T)]["dr"] - base_dr > -0.02 for T in Ts[1:])
    worst = max(Ts[1:], key=lambda T: k2[str(T)]["proxy_alarms"])
    side = {"T": worst,
            "full_pipeline_ratio": k2[str(worst)]["alarms"] / max(k2["None"]["alarms"], 1),
            "behavioral_proxy_ratio": k2[str(worst)]["proxy_alarms"]
                                      / max(k2["None"]["proxy_alarms"], 1),
            "note": "Memo/makale (17x) davranissal-dal PROXY'sini olctu; tam uretim "
                    "hattinda (immature/early_attack kapilari dahil) artis cok daha "
                    "kucuk. Makaledeki 'live false-alarm rate rose 17x' ifadesi "
                    "proxy'ye aittir."}
    result = {
        "gate": "GATE-AT-0 active flow timeout (zero-training)",
        "paper_table": "dataset_generalization Table 1",
        "K1_cicids_row_change": k1, "K2_live_ood": k2, "K3_slow_dos": k3,
        "criteria": {"K1 (<5% rows change)": k1_pass, "K2 (live OOD p95 < 21)": k2_pass,
                     "K3 (slow-DoS DR drop < 2pt)": k3_pass},
        "side_effect_live_alarm": side,
        "verdict": "FAIL -- do not run GATE-AT-1" if not (k1_pass and k2_pass and k3_pass)
                   else "PASS",
        "memo_reference": MEMO,
        "provenance": common.provenance(("",), [LIVE_PCAP]),
    }
    common.write_json("gate_at0_active_timeout.json", result)
    print(result["criteria"], result["side_effect_live_alarm"], result["verdict"])


if __name__ == "__main__":
    main()
