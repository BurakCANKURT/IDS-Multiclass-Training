# -*- coding: utf-8 -*-
"""
pcap_validation_pilot_v2.py
============================
pcap_validation_pilot.py'nin BIREBIR aynisi; SADECE MainRF/MainLSTM yerine
MainRF_v2/MainLSTM_v2 (streaming_training_dataset.csv ile, UniversalRealTime-
FlowTable'in URETTIGI incremental ozelliklerle yeniden egitilmis) kullanir.

Dispatcher mantigi (update_volumetric/update_progressive/esikler/SYN_BURST_MIN/
LSTM_WEIGHT/COUNTER_WEIGHT) DEGISMEDI -> v1 vs v2 karsilastirmasi SADECE
model agirliklarinin etkisini izole eder. Ayni 12 batch, ayni ground truth.
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

PROJECT_DIR = v1.PROJECT_DIR
MAIN_RF_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainRF_v2.joblib")
MAIN_LSTM_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainLSTM_v2.keras")
BUNDLE_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainRF_v2__preprocess.joblib")

OUT_CSV = os.path.join(PROJECT_DIR, "results", "pcap_validation_results_v2.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "results", "pcap_validation_summary_v2.json")


def load_models_v2():
    import joblib
    import tensorflow as tf
    print("[*] v2 modeller yukleniyor (bir kez)...")
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

    # NOT: tf.keras.models.load_model() ile donen modelin call() metodu,
    # v1'in build_and_load_dynamic_lstm (Sequential + set_weights, hic
    # load_model yok) yoluyla insa edilenden cok daha yavas calisiyor
    # (her cagrida ~60-90ms vs ~3-4ms). Burada agirliklari cikarip TAZE
    # bir Sequential'a yukluyoruz -> v1 ile birebir ayni insa yolu.
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

    print(f"    RF classes_={cfg['RF_CLASSES']} | vol_thr={cfg['VOLUMETRIC_THRESHOLD']} "
          f"| BENIGN_INDEX={cfg['BENIGN_INDEX']} | raw_cols={len(cfg['RAW_COLUMNS'])} "
          f"| lstm_in={n_feat_lstm}")
    return cfg


def main():
    if not os.path.exists(v1.TSHARK):
        print(f"[HATA] tshark bulunamadi: {v1.TSHARK}")
        return

    cfg = load_models_v2()
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

        flow_table = v1.UniversalRealTimeFlowTable()
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
    print("GENEL (v2, paket-bazli, tum pilot batch'ler):")
    for k, val in overall.items():
        print(f"  {k:20s}: {val}")
    print(f"\n[+] Paket-bazli sonuc : {OUT_CSV}")
    print(f"[+] Ozet (JSON)       : {OUT_JSON}")


if __name__ == "__main__":
    main()
