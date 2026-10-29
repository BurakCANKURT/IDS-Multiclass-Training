# -*- coding: utf-8 -*-
"""
convert_multiclass_committee_to_onnx.py
=========================================
R26 (davranissal-dal konsey vetosu, Faz 1): MulticlassRF + MulticlassHGB'yi
ONNX'e cevirir + held-out test setinde sayisal esdegerlik/latency olcer.
convert_rf_to_onnx.py (binary MainRF_v5) ile AYNI strateji, 7-sinifli cikisa
genellenmis (zipmap=False -> ham [n,7] olasilik matrisi; binary'deki
skl2onnx P0-sutunu kusuru MULTICLASS'TA GORULMEDI -- asagida her iki sutun
da dogrudan karsilastirilir, workaround GEREKMEDI).

MODEL CIKTISI (predict_proba) DEGISMEMELI; committee_dissent vetosu SADECE
P(BENIGN) esigine bakiyor, bu yuzden regresyon kapisi P(BENIGN) sutunundaki
farka odaklanir (tum satirlarda, veto-tetikleyen alt-kumede ayrica).

Kullanim:
    python convert_multiclass_committee_to_onnx.py
"""
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC_DIR = os.path.join(PROJECT_DIR, "multiclass_training")
MODELS_DIR = os.path.join(MC_DIR, "models")
DATA_CSV = os.path.join(MC_DIR, "data", "multiclass_streaming_dataset.csv")
CLASSES_JSON = os.path.join(MC_DIR, "data", "streaming_label_classes.json")

RF_JOBLIB = os.path.join(MODELS_DIR, "MulticlassRF.joblib")
HGB_JOBLIB = os.path.join(MODELS_DIR, "MulticlassHGB.joblib")
RF_ONNX = os.path.join(MODELS_DIR, "MulticlassRF.onnx")
HGB_ONNX = os.path.join(MODELS_DIR, "MulticlassHGB.onnx")

sys.path.insert(0, MC_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "analysis"))
from train_multiclass_models import batch_level_split
from pcap_validation_pilot_v5 import RAW_COLUMNS_V5

N_FEATURES = 41
VETO_THRESHOLD = 0.95


def convert(model, out_path, n_classes):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
        target_opset=12,
    )
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"[+] ONNX kaydedildi: {out_path} ({size_kb:.0f} KB)")


def make_session(onnx_path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    prob_name = sess.get_outputs()[1].name
    return sess, in_name, prob_name


def onnx_predict_proba(sess, in_name, prob_name, X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    out = np.asarray(sess.run([prob_name], {in_name: X})[0], dtype=np.float64)
    return out


def equivalence(name, model, X, benign_idx):
    sess, in_name, prob_name = make_session(
        RF_ONNX if name == "RF" else HGB_ONNX
    )
    p_jl = model.predict_proba(X)
    p_ox = onnx_predict_proba(sess, in_name, prob_name, X)

    diff = np.abs(p_jl - p_ox)
    max_d = float(diff.max())
    mean_d = float(diff.mean())

    argmax_jl = p_jl.argmax(1)
    argmax_ox = p_ox.argmax(1)
    flips = int((argmax_jl != argmax_ox).sum())

    ben_jl = p_jl[:, benign_idx]
    ben_ox = p_ox[:, benign_idx]
    veto_jl = ben_jl >= VETO_THRESHOLD
    veto_ox = ben_ox >= VETO_THRESHOLD
    veto_flips = int((veto_jl != veto_ox).sum())

    print(f"\n=== {name} SAYISAL ESDEGERLIK (n={X.shape[0]}) ===")
    print(f"  max |predict_proba farki|         : {max_d:.3e}")
    print(f"  mean |predict_proba farki|        : {mean_d:.3e}")
    print(f"  argmax karar degisimi             : {flips}")
    print(f"  P(BENIGN)>={VETO_THRESHOLD} veto-karar degisimi : {veto_flips}")
    ok = (veto_flips == 0)
    print(f"  REGRESYON KAPISI (veto karari)     : {'GECTI' if ok else 'KALDI (entegre etme!)'}")
    return ok, max_d, veto_flips


def benchmark(name, model, X, n_iter=500):
    sess, in_name, prob_name = make_session(RF_ONNX if name == "RF" else HGB_ONNX)
    rows = X[: min(n_iter, X.shape[0])]
    model.predict_proba(rows[:1])
    onnx_predict_proba(sess, in_name, prob_name, rows[:1])

    jl_t = []
    for r in rows:
        x1 = r.reshape(1, -1)
        t0 = time.perf_counter()
        model.predict_proba(x1)
        jl_t.append((time.perf_counter() - t0) * 1e3)

    ox_t = []
    for r in rows:
        x1 = r.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter()
        onnx_predict_proba(sess, in_name, prob_name, x1)
        ox_t.append((time.perf_counter() - t0) * 1e3)

    jl_t = np.array(jl_t); ox_t = np.array(ox_t)
    print(f"\n=== {name} LATENCY (tek-satir, ms) ===")
    print(f"  joblib  medyan={np.median(jl_t):.3f}  ort={jl_t.mean():.3f}")
    print(f"  ONNX    medyan={np.median(ox_t):.3f}  ort={ox_t.mean():.3f}")
    speed = np.median(jl_t) / max(np.median(ox_t), 1e-9)
    print(f"  HIZLANMA (medyan): {speed:.1f}x")
    return float(np.median(jl_t)), float(np.median(ox_t)), float(speed)


def main():
    print("[*] Modeller yukleniyor...")
    rf = joblib.load(RF_JOBLIB)
    hgb = joblib.load(HGB_JOBLIB)
    bundle = joblib.load(os.path.join(MODELS_DIR, "MulticlassRF_bundle.joblib"))
    classes = bundle["classes"]
    benign_idx = bundle["benign_index"]
    scaler = bundle["pipeline"].named_steps["scaler"]
    n_classes = len(classes)
    print(f"    RF n_est={rf.n_estimators}  HGB n_iter={hgb.max_iter}  classes={classes}")

    print("\n[*] ONNX'e ceviriliyor...")
    convert(rf, RF_ONNX, n_classes)
    convert(hgb, HGB_ONNX, n_classes)

    print("\n[*] Held-out test setinden dogrulama satirlari toplaniyor (gercek egitim split'i)...")
    df = pd.read_csv(DATA_CSV)
    _, df_test = batch_level_split(df, test_ratio=0.2, random_state=42)
    X = scaler.transform(df_test[RAW_COLUMNS_V5].values.astype("float64"))
    print(f"    {X.shape[0]:,} satir x {X.shape[1]} ozellik")

    ok_rf, max_d_rf, vf_rf = equivalence("RF", rf, X, benign_idx)
    ok_hgb, max_d_hgb, vf_hgb = equivalence("HGB", hgb, X, benign_idx)

    benchmark("RF", rf, X)
    benchmark("HGB", hgb, X)

    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)
    ok = ok_rf and ok_hgb
    print(f"  RF  : max|d|={max_d_rf:.2e}  veto-karar-degisimi={vf_rf}")
    print(f"  HGB : max|d|={max_d_hgb:.2e}  veto-karar-degisimi={vf_hgb}")
    print(f"  ENTEGRASYON ONERISI: {'EVET (regresyon kapisi gecti)' if ok else 'HAYIR (sapma var, entegre etme)'}")


if __name__ == "__main__":
    main()
