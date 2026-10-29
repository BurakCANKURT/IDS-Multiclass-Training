# -*- coding: utf-8 -*-
"""
pcap_validation_pilot_v5.py
============================
v5.0 minimal ozellik seti: 32 -> 41 raw_columns.
  - "Destination Port" (raw_columns[0]) -> 7 kategorik PortBucket_* one-hot
    (_port_bucket: ftp/ssh/http(8080 dahil)/https/dns/other_wk/ephemeral).
    "Destination Port" key'i master_features icinde KORUNUR (dispatcher
    dst_port ciktisi icin), sadece RAW_COLUMNS_V5'e GIRMEZ.
  - olu "Active Max"/"Active Min" (hep 0.0, pcap_validation_pilot.py:390) ->
    SYN/FIN/RST Flag Count + Down/Up Ratio + Fwd/Bwd Byte Ratio (gercek
    davranissal sinyaller).

UniversalRealTimeFlowTableV5(v1.UniversalRealTimeFlowTable): extract_universal_features()
once super()'i cagirir (32 v1 ozelligi), ardindan 12 yeni ozelligi ekler.

load_models_v5(): pcap_validation_pilot_v4.load_models_v4() ile AYNI yapi;
MainRF_v5/MainLSTM_v5/MainRF_v5__preprocess dosyalarini yukler, raw_columns ve
n_feat_lstm == 41 assert eder.

main(): Wednesday 12-batch pilot (v1.ATTACK_BATCHES/CLEAN_BATCHES), v1.gt_label/
dispatch_packet/confusion/metrics_from_conf ile v4 ile AYNI -- sadece yeni
ozellik vektoru + yeni model agirliklarinin etkisini izole eder.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

import pcap_validation_pilot as v1
from pcap_validation_pilot import _parse_tcp_flags

PROJECT_DIR = v1.PROJECT_DIR
MAIN_RF_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainRF_v5.joblib")
MAIN_LSTM_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainLSTM_v5.keras")
BUNDLE_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainRF_v5__preprocess.joblib")

OUT_CSV = os.path.join(PROJECT_DIR, "results", "pcap_validation_results_v5.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "results", "pcap_validation_summary_v5.json")


# ---------------------------------------------------------------------------
# v5.0: PORT BUCKETING
# ---------------------------------------------------------------------------
PORT_BUCKETS = ["ftp", "ssh", "http", "https", "dns", "other_wk", "ephemeral"]


def _port_bucket(port):
    p = int(port)
    if p == 21:
        return "ftp"
    if p == 22:
        return "ssh"
    if p in (80, 8080):
        return "http"
    if p == 443:
        return "https"
    if p == 53:
        return "dns"
    if p < 1024:
        return "other_wk"
    return "ephemeral"


# ---------------------------------------------------------------------------
# v5.0: 41 raw_columns -- "Destination Port" + olu "Active Max"/"Active Min"
# cikarildi (32-3=29); 7 PortBucket_* one-hot + SYN/FIN/RST Flag Count +
# Down/Up Ratio + Fwd/Bwd Byte Ratio eklendi (29+12=41).
# ---------------------------------------------------------------------------
RAW_COLUMNS_V5 = [f"PortBucket_{b}" for b in PORT_BUCKETS] + [
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Bytes",
    "Flow IAT Mean",
    "Fwd IAT Max",
    "Bwd IAT Total",
    "Bwd IAT Max",
    "Bwd Packets/s",
    "PSH Flag Count",
    "ACK Flag Count",
    "Max Packet Length",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "Subflow Bwd Packets",
    "SYN Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "Down/Up Ratio",
    "Fwd/Bwd Byte Ratio",
    "Flow IAT Max",
    "Flow Bytes/s",
    "Flow Packets/s",
]
assert len(RAW_COLUMNS_V5) == 41, len(RAW_COLUMNS_V5)


# ---------------------------------------------------------------------------
# UniversalRealTimeFlowTableV5
# ---------------------------------------------------------------------------
class UniversalRealTimeFlowTableV5(v1.UniversalRealTimeFlowTable):
    """v1 + v5.0: PortBucket_* one-hot + SYN/FIN/RST Flag Count + Down/Up Ratio
    + Fwd/Bwd Byte Ratio. "Destination Port" key korunur (dispatcher dst_port
    cikisi icin), RAW_COLUMNS_V5'te degil."""

    def __init__(self):
        super().__init__()
        self._flag_counts = {}

    def extract_universal_features(self, packet):
        master_features, flow_key = super().extract_universal_features(packet)
        if master_features is None:
            return None, None

        flags_int = _parse_tcp_flags(packet.get("tcp.flags"))
        st = self._flag_counts.get(flow_key)
        if st is None:
            st = {"syn": 0, "fin": 0, "rst": 0}
            self._flag_counts[flow_key] = st
        if flags_int & 0x02:
            st["syn"] += 1
        if flags_int & 0x01:
            st["fin"] += 1
        if flags_int & 0x04:
            st["rst"] += 1

        flow = self.flows[flow_key]
        fwd_pkts = flow["fwd_packets"]
        bwd_pkts = flow["bwd_packets"]
        fwd_sum = sum(flow["fwd_lengths"])
        bwd_sum = sum(flow["bwd_lengths"])

        bucket = _port_bucket(master_features["Destination Port"])
        for b in PORT_BUCKETS:
            master_features[f"PortBucket_{b}"] = 1.0 if b == bucket else 0.0

        master_features["SYN Flag Count"] = float(st["syn"])
        master_features["FIN Flag Count"] = float(st["fin"])
        master_features["RST Flag Count"] = float(st["rst"])
        master_features["Down/Up Ratio"] = float(bwd_pkts / max(fwd_pkts, 1))
        master_features["Fwd/Bwd Byte Ratio"] = float(fwd_sum / max(bwd_sum, 1))
        return master_features, flow_key


# ---------------------------------------------------------------------------
# load_models_v5 (pcap_validation_pilot_v4.load_models_v4 ile ayni yapi)
# ---------------------------------------------------------------------------
def load_models_v5():
    import joblib
    import tensorflow as tf
    print("[*] v5 modeller yukleniyor (bir kez)...")
    bundle = joblib.load(BUNDLE_PATH)
    cfg = {
        "PIPELINE": bundle["pipeline"],
        "RAW_COLUMNS": bundle["raw_columns"],
        "CLASSES_LIST": bundle.get("classes", ["BENIGN", "ATTACK"]),
        "VOLUMETRIC_FEATURE": bundle.get("volumetric_feature", "Flow Packets/s"),
        "VOLUMETRIC_THRESHOLD": bundle.get("volumetric_threshold", float("inf")),
    }
    cfg["BENIGN_INDEX"] = bundle.get("benign_index", 0)

    main_rf = joblib.load(MAIN_RF_PATH)
    if isinstance(main_rf, dict) and "model" in main_rf:
        main_rf = main_rf["model"]
    cfg["main_rf"] = main_rf
    cfg["RF_CLASSES"] = list(getattr(main_rf, "classes_", [0, 1]))
    cfg["RF_BENIGN_POS"] = cfg["RF_CLASSES"].index(0) if 0 in cfg["RF_CLASSES"] else 0
    # PERF (Strateji-1): ONNX RF hizlandirma (joblib RF.predict_proba ~36ms ->
    # ONNX ~0.04ms). v1.dispatch_packet rf_predict_proba ile bu session'i kullanir.
    cfg["rf_onnx"] = v1._try_load_rf_onnx(MAIN_RF_PATH)
    # (R28/O2.1) konsey vetosunun ikinci uyesi (HistGB). Yoksa konsey no-op.
    cfg["hgb_onnx"] = v1._try_load_hgb_onnx(
        os.path.join(os.path.dirname(MAIN_RF_PATH), "MainHGB_v5.joblib"))

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

    loaded = tf.keras.models.load_model(MAIN_LSTM_PATH)
    n_feat_lstm = loaded.input_shape[-1]
    weights = loaded.get_weights()
    del loaded

    fresh = Sequential([
        Input(shape=(1, n_feat_lstm)),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    fresh.set_weights(weights)
    cfg["main_lstm"] = fresh

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1, n_feat_lstm], dtype=tf.float32)])
    def _lstm_fast(x):
        return cfg["main_lstm"](x, training=False)

    cfg["lstm_fast"] = _lstm_fast
    cfg["n_feat_lstm"] = n_feat_lstm

    assert len(cfg["RAW_COLUMNS"]) == 41, f"RAW_COLUMNS={len(cfg['RAW_COLUMNS'])}, expected 41"
    assert n_feat_lstm == 41, f"n_feat_lstm={n_feat_lstm}, expected 41"

    print(f"    RF classes_={cfg['RF_CLASSES']} | vol_thr={cfg['VOLUMETRIC_THRESHOLD']} "
          f"| BENIGN_INDEX={cfg['BENIGN_INDEX']} | raw_cols={len(cfg['RAW_COLUMNS'])} "
          f"| lstm_in={n_feat_lstm}")
    return cfg


# ---------------------------------------------------------------------------
# main: Wednesday 12-batch pilot (v4 ile AYNI batch'ler/gt/dispatch)
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(v1.TSHARK):
        print(f"[HATA] tshark bulunamadi: {v1.TSHARK}")
        return

    cfg = load_models_v5()
    selected = [(b, "attack") for b in v1.ATTACK_BATCHES] + [(b, "clean") for b in v1.CLEAN_BATCHES]

    all_rows = []
    batch_summ = []
    glob_true, glob_pred = [], []

    for idx, expected in selected:
        pcap = v1.find_batch(idx)
        if not pcap:
            print(f"[!] batch {idx} bulunamadi, atlaniyor.")
            continue
        print(f"\n[batch {idx} | beklenen={expected}] {os.path.basename(pcap)}", flush=True)
        packets = v1.read_pcap(pcap)
        print(f"    {len(packets)} paket okundu, dispatcher calisiyor...", flush=True)

        flow_table = UniversalRealTimeFlowTableV5()
        y_true, y_pred, risks = [], [], []
        n_rf = n_lstm = 0

        for pkt in packets:
            gt = v1.gt_label(pkt)
            res = v1.dispatch_packet(pkt, flow_table, cfg)
            alert = 1 if res["risk_score"] >= v1.ALERT_THRESHOLD else 0
            y_true.append(gt)
            y_pred.append(alert)
            risks.append(res["risk_score"])
            if res["model"] == "RF":
                n_rf += 1
            else:
                n_lstm += 1
            all_rows.append({
                "batch": idx, "expected": expected,
                "src_ip": res["src_ip"], "dst_ip": res["dst_ip"],
                "dst_port": res["dst_port"], "model": res["model"],
                "is_volumetric": res["is_volumetric"],
                "risk_score": round(res["risk_score"], 4),
                "gt_attack": gt, "alert": alert,
            })

        glob_true += y_true
        glob_pred += y_pred
        tp, fp, tn, fn = v1.confusion(y_true, y_pred)
        n_alert = sum(y_pred)
        n_attack = sum(y_true)
        risks_np = np.array(risks) if risks else np.array([0.0])
        bs = {
            "batch": idx, "expected": expected, "pcap": os.path.basename(pcap),
            "total_pkts": len(packets),
            "gt_attack_pkts": n_attack, "gt_benign_pkts": len(packets) - n_attack,
            "alerts": n_alert,
            "mean_risk": round(float(risks_np.mean()), 4),
            "max_risk": round(float(risks_np.max()), 4),
            "n_RF": n_rf, "n_LSTM": n_lstm,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        }
        bs.update(v1.metrics_from_conf(tp, fp, tn, fn))
        batch_summ.append(bs)
        print(f"    pkts={len(packets)} gt_attack={n_attack} alerts={n_alert} "
              f"RF={n_rf} LSTM={n_lstm} | DR={bs['DR_recall_attack']} FAR={bs['FAR']} "
              f"meanRisk={bs['mean_risk']} maxRisk={bs['max_risk']}", flush=True)

    gtp, gfp, gtn, gfn = v1.confusion(glob_true, glob_pred)
    overall = {
        "total_pkts": len(glob_true),
        "gt_attack_pkts": sum(glob_true), "gt_benign_pkts": len(glob_true) - sum(glob_true),
        "alerts": sum(glob_pred),
        "TP": gtp, "FP": gfp, "TN": gtn, "FN": gfn,
    }
    overall.update(v1.metrics_from_conf(gtp, gfp, gtn, gfn))

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "batches": batch_summ}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("GENEL (v5, Wednesday pilot, paket-bazli, tum batch'ler):")
    for k, val in overall.items():
        print(f"  {k:20s}: {val}")
    print(f"\n[+] Paket-bazli sonuc : {OUT_CSV}")
    print(f"[+] Ozet (JSON)       : {OUT_JSON}")


if __name__ == "__main__":
    main()
