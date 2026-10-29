# -*- coding: utf-8 -*-
"""
verify_lstm_fastpath.py
========================
LSTM "fast-path" (tf.function icinde dogrudan `model(x, training=False)`)
ile eski Keras `.predict()` yolunun SAYISAL ESDEGERLIGINI kanitlar.

Gerekce: `.predict()` tek-paket cagrilarinda paket basina ~86-90ms
Python/callback/DataAdapter overhead'i ekliyordu; canli hatta bu, detektoru
~15 paket/sn ile sinirlayip yuk altinda %97-99.6 girdi kaybina yol aciyordu
(orkestrator kuyruk sayaci ile olculdu). Fast-path ayni forward-pass
grafigini cagirir, dolayisiyla cikti DEGISMEMELIDIR.

Bu betik, RF->ONNX gecisinde uygulanan REGRESYON KAPISININ aynisini uygular:
"hizlandi" demek yetmez, "hem hizlandi HEM sonuc ayni" kanitlanmalidir.

Olcut (her iki model icin ayri):
  1. max|fark| ve mean|fark| -- tolerans TOL (varsayilan 1e-5)
  2. ALERT_THRESHOLD=0.4 civari KARAR degisimi (alert/no-alert) = 0 olmali
  3. argmax (multiclass sinif secimi) degisimi = 0 olmali
  4. tek-satir latency: eski yol vs yeni yol

Kullanim:
    python analysis/verify_lstm_fastpath.py
"""
import os
import sys
import time
import statistics
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "multiclass_training"))

import tensorflow as tf
from offline_test_multiclass import build_and_load_dynamic_lstm

TOL = float(os.environ.get("LSTM_FASTPATH_TOL", "1e-5"))
ALERT_THRESHOLD = 0.4
N_SAMPLES = int(os.environ.get("LSTM_FASTPATH_N", "300"))

MODELS = [
    ("BINARY  MainLSTM_v5",
     os.path.join(PROJECT_DIR, "saved_models", "MainLSTM_v5.keras")),
    ("MULTICLASS MulticlassLSTM",
     os.path.join(PROJECT_DIR, "multiclass_training", "models", "MulticlassLSTM.keras")),
]


def real_feature_sample(n_feat, n):
    """Mumkunse GERCEK uretim ozellik dagilimindan ornek al (rastgele
    gurultu, modelin uretimde gordugu bolgeyi temsil etmez). Bulunamazsa
    normal dagilima duser."""
    for rel in [("data", "streaming_training_dataset_v5.csv"),
                ("multiclass_training", "data", "multiclass_streaming_dataset.csv")]:
        p = os.path.join(PROJECT_DIR, *rel)
        if not os.path.exists(p):
            continue
        try:
            import pandas as pd
            df = pd.read_csv(p, nrows=n * 3)
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] >= n_feat and len(num) >= n:
                X = num.iloc[:n, :n_feat].to_numpy(dtype="float32")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                return X, f"gercek ozellikler ({os.path.basename(p)})"
        except Exception:
            pass
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, n_feat)).astype("float32"), "sentetik (normal)"


def bench(fn, X, n_warm=10):
    for i in range(min(n_warm, len(X))):
        fn(X[i:i + 1])
    ts = []
    for i in range(len(X)):
        t0 = time.perf_counter()
        fn(X[i:i + 1])
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main():
    print(f"Tolerans: {TOL}   ornek sayisi: {N_SAMPLES}\n")
    all_ok = True

    for label, path in MODELS:
        print("=" * 74)
        print(label)
        print("=" * 74)
        if not os.path.exists(path):
            print(f"  [!] model bulunamadi: {path}")
            all_ok = False
            continue

        model = build_and_load_dynamic_lstm(path)
        n_feat = model.input_shape[-1]
        n_out = model.output_shape[-1]
        X2d, src = real_feature_sample(n_feat, N_SAMPLES)
        X = X2d.reshape(len(X2d), 1, n_feat)
        print(f"  girdi: {X.shape}  ({src}), cikis boyutu={n_out}")

        fast = tf.function(lambda t: model(t, training=False), reduce_retracing=True)

        def old_path(x):
            return model.predict(x, verbose=0)

        def new_path(x):
            return np.asarray(fast(tf.constant(x, dtype=tf.float32)))

        old_out = np.concatenate([old_path(X[i:i + 1]) for i in range(len(X))], axis=0)
        new_out = np.concatenate([new_path(X[i:i + 1]) for i in range(len(X))], axis=0)

        diff = np.abs(old_out - new_out)
        max_d, mean_d = float(diff.max()), float(diff.mean())
        print(f"  max|fark|  = {max_d:.3e}")
        print(f"  mean|fark| = {mean_d:.3e}")

        # (2) karar degisimi -- ikili cikis icin p_attack, multiclass icin
        # 1 - p_benign uzerinden ALERT_THRESHOLD karsilastirmasi.
        if n_out == 1:
            old_risk, new_risk = old_out[:, 0], new_out[:, 0]
        else:
            old_risk, new_risk = 1.0 - old_out[:, 0], 1.0 - new_out[:, 0]
        flips = int(np.sum((old_risk >= ALERT_THRESHOLD) != (new_risk >= ALERT_THRESHOLD)))
        print(f"  ALERT_THRESHOLD({ALERT_THRESHOLD}) karar degisimi = {flips}")

        # (3) argmax degisimi (multiclass sinif secimi)
        arg_flips = 0
        if n_out > 1:
            arg_flips = int(np.sum(np.argmax(old_out, axis=1) != np.argmax(new_out, axis=1)))
            print(f"  argmax (sinif) degisimi = {arg_flips}")

        # (4) latency
        med_old = bench(old_path, X[:120])
        med_new = bench(new_path, X[:120])
        print(f"  latency: .predict()={med_old:.3f}ms  fast-path={med_new:.4f}ms  "
              f"-> {med_old / max(med_new, 1e-9):.0f}x")

        ok = (max_d <= TOL) and (flips == 0) and (arg_flips == 0)
        all_ok &= ok
        print(f"  REGRESYON KAPISI: {'GECTI' if ok else 'KALDI (entegre etme!)'}\n")

    print("=" * 74)
    print(f"GENEL SONUC: {'TUM MODELLER GECTI' if all_ok else 'EN AZ BIR MODEL KALDI'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
