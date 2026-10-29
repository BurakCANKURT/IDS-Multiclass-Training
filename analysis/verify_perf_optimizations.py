# -*- coding: utf-8 -*-
"""
verify_perf_optimizations.py
=============================
Konu-1 profillemesinden cikan UC HIZ optimizasyonunun SAYISAL ESDEGERLIGINI
kanitlar. Standart: `analysis/verify_lstm_fastpath.py` / RF->ONNX gecisiyle
AYNI ("hizlandi" demek yetmez, "hem hizlandi HEM sonuc ayni" kanitlanmali).

Kalemler:
  1. Tek-satir pandas DataFrame -> dogrudan numpy satiri (`_feature_row`)
  2. `tf.constant(...)` donusumu yerine onceden-ayrilmis/yeniden-kullanilan yol
  3. OOD 1-NN'in yalnizca karara etki eden dalda hesaplanmasi

Her kalem AYRI test edilir; biri kalirsa digerleri etkilenmez.
Tolerans varsayilan 0.0 (BIT-BIREBIR) -- bu uc degisiklik saf yeniden
duzenleme oldugu icin float farki bile BEKLENMEZ.

Kullanim:
    python analysis/verify_perf_optimizations.py
"""
import os
import sys
import time
import statistics
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

TOL = float(os.environ.get("PERF_VERIFY_TOL", "0.0"))
N = int(os.environ.get("PERF_VERIFY_N", "2000"))


def load_binary_cfg():
    """Ikili hattin PIPELINE + RAW_COLUMNS'unu, process_pcap.py'yi calistirmadan
    (Kafka/stdin dongusune girmeden) dogrudan bundle'dan yukler."""
    import joblib
    bundle = joblib.load(os.path.join(PROJECT_DIR, "saved_models",
                                       "MainRF_v5__preprocess.joblib"))
    return bundle["pipeline"], bundle["raw_columns"]


def sample_feature_dicts(raw_columns, n):
    """GERCEK uretim ozellik dagilimindan ornek master_features sozlukleri."""
    p = os.path.join(PROJECT_DIR, "data", "streaming_training_dataset_v5.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, nrows=n)
        cols = [c for c in raw_columns if c in df.columns]
        out = []
        for _, r in df.iterrows():
            d = {c: float(r[c]) for c in cols}
            out.append(d)
        return out, f"gercek ({os.path.basename(p)}, {len(cols)}/{len(raw_columns)} kolon eslesti)"
    rng = np.random.default_rng(0)
    return ([{c: float(v) for c, v in zip(raw_columns, rng.normal(size=len(raw_columns)))}
             for _ in range(n)], "sentetik")


def bench(fn, args, n_warm=20):
    for a in args[:n_warm]:
        fn(a)
    ts = []
    for a in args:
        t0 = time.perf_counter()
        fn(a)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def item1(pipeline, raw_columns, feats):
    print("=" * 74)
    print("KALEM 1 -- tek-satir DataFrame -> dogrudan numpy satiri")
    print("=" * 74)

    def old(mf):
        row_df = pd.DataFrame(
            [{col: float(mf.get(col, 0.0)) for col in raw_columns}],
            columns=raw_columns,
        )
        return row_df.values

    def new(mf):
        return np.array([[float(mf.get(col, 0.0)) for col in raw_columns]],
                        dtype=np.float64)

    max_d = 0.0
    for mf in feats:
        a, b = old(mf), new(mf)
        if a.shape != b.shape:
            print(f"  SEKIL FARKI: {a.shape} vs {b.shape}"); return False
        max_d = max(max_d, float(np.abs(a - b).max()))
    print(f"  transform ONCESI dizi   max|fark| = {max_d:.3e}")

    # transform CIKTISI da karsilastirilir (asil onemli olan bu)
    max_t = 0.0
    for mf in feats[:500]:
        ta = pipeline.transform(old(mf))
        tb = pipeline.transform(new(mf))
        max_t = max(max_t, float(np.abs(ta - tb).max()))
    print(f"  PIPELINE.transform sonrasi max|fark| = {max_t:.3e}")

    mo, mn = bench(old, feats), bench(new, feats)
    print(f"  latency: DataFrame={mo:.4f}ms  numpy={mn:.4f}ms  -> {mo / max(mn, 1e-9):.1f}x "
          f"(kazanc {mo - mn:.4f} ms/paket)")
    ok = (max_d <= TOL) and (max_t <= TOL)
    print(f"  REGRESYON KAPISI: {'GECTI' if ok else 'KALDI'}\n")
    return ok


def item2(feats, raw_columns):
    print("=" * 74)
    print("KALEM 2 -- tf.constant donusumu")
    print("=" * 74)
    import tensorflow as tf
    sys.path.insert(0, os.path.join(PROJECT_DIR, "multiclass_training"))
    from offline_test_multiclass import build_and_load_dynamic_lstm
    mp = os.path.join(PROJECT_DIR, "saved_models", "MainLSTM_v5.keras")
    if not os.path.exists(mp):
        print("  [!] model yok, atlaniyor\n"); return None
    model = build_and_load_dynamic_lstm(mp)
    nf = model.input_shape[-1]
    X = [np.random.default_rng(i).normal(size=(1, 1, nf)).astype("float32") for i in range(400)]
    fast = tf.function(lambda t: model(t, training=False), reduce_retracing=True)

    def old(x):
        return np.asarray(fast(tf.constant(x, dtype=tf.float32)))

    # Yeni yol: numpy'i dogrudan ver (TF zaten otomatik cevirir, ek kopya yok)
    def new(x):
        return np.asarray(fast(x))

    max_d = max(float(np.abs(old(x) - new(x)).max()) for x in X[:200])
    print(f"  max|fark| = {max_d:.3e}")
    mo, mn = bench(old, X), bench(new, X)
    print(f"  latency: tf.constant={mo:.4f}ms  dogrudan={mn:.4f}ms  "
          f"-> kazanc {mo - mn:.4f} ms/paket")
    ok = max_d <= TOL
    print(f"  REGRESYON KAPISI: {'GECTI' if ok else 'KALDI'}\n")
    return ok


def item3():
    print("=" * 74)
    print("KALEM 3 -- OOD 1-NN yalnizca karara etki eden dalda")
    print("=" * 74)
    print("  Bu kalem bir HESAP degil, bir ATLAMA optimizasyonudur:")
    print("  volumetrik dalda OOD mesafesi (R29/O4) SALT GOZLEMdir --")
    print("  is_ood daima False kalir, karara ETKI ETMEZ. Dolayisiyla")
    print("  atlanmasi status/risk_score'u tanim geregi degistiremez;")
    print("  degisen tek sey Mongo'daki ood_distance alaninin volumetrik")
    print("  satirlarda None olmasidir (R29/O4 gozlem verisi KAYBOLUR).")
    print("  -> Sayisal esdegerlik testi UYGULANAMAZ; karar-esdegerligi")
    print("     yapisaldir. Gozlem kaybi nedeniyle KULLANICI ONAYINA birakildi.\n")
    return None


def main():
    pipeline, raw_columns = load_binary_cfg()
    feats, src = sample_feature_dicts(raw_columns, N)
    print(f"Tolerans={TOL} (0.0 = bit-birebir)  ornek={len(feats)}  kaynak: {src}\n")
    r1 = item1(pipeline, raw_columns, feats)
    r2 = item2(feats, raw_columns)
    r3 = item3()
    print("=" * 74)
    for name, r in [("KALEM 1 (DataFrame)", r1), ("KALEM 2 (tf.constant)", r2),
                    ("KALEM 3 (OOD atlama)", r3)]:
        print(f"  {name:26s}: {'GECTI' if r else ('ATLANDI/ONAY BEKLIYOR' if r is None else 'KALDI')}")
    return 0 if (r1 and (r2 is not False)) else 1


if __name__ == "__main__":
    sys.exit(main())
