# -*- coding: utf-8 -*-
"""
killgates/common.py
===================
Kill-gate harness'lerinin ORTAK altyapisi.

Buradaki her fonksiyon uretim kodunu IMPORT eder, kopyalamaz:
  * korpus bolmesi   -> train_multiclass_models.batch_level_split (egitimdeki birebir)
  * gercek-dunya     -> offline_test_multiclass.run_offline + validate_realworld_pcaps
                        registry'si (bkz. realworld.py)

Hicbir fonksiyon uretim model dosyasina, results/ dizinine veya egitim CSV'sine
YAZMAZ. Tum ciktilar analysis/killgates/out/ altina gider.
"""

import hashlib
import json
import math
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
MC_DIR = os.path.join(PROJECT_DIR, "multiclass_training")
MODEL_DIR = os.path.join(MC_DIR, "models")
DATA_CSV = os.path.join(MC_DIR, "data", "multiclass_streaming_dataset.csv")
CLASSES_JSON = os.path.join(MC_DIR, "data", "streaming_label_classes.json")
OUT_DIR = os.path.join(THIS_DIR, "out")

for _p in (MC_DIR, os.path.join(PROJECT_DIR, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 42


# ---------------------------------------------------------------------------
# Istatistik
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson skor araligi (k basari / n deneme). Makaledeki Tablo 4/5 formati."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"),
            "ci95": [lo, hi]}


def cluster_bootstrap(clusters, hits, B=2000, seed=SEED, hits_b=None, stat="rate"):
    """Kume (akis/oturum/batch) duzeyinde bootstrap.

    Satirlar bagimsiz DEGIL (ayni akisin paketleri); kumeler yeniden orneklenir.
    hits_b verilirse AYNI kumeler uzerinde eslestirilmis ikinci kol: stat
    'diff' (b - a) ya da 'ratio' (b / a) icin aralik doner.
    """
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"c": clusters, "a": np.asarray(hits, dtype=float)})
    if hits_b is not None:
        d["b"] = np.asarray(hits_b, dtype=float)
    g = d.groupby("c").agg(["sum", "count"]) if hits_b is None else d.groupby("c").sum()
    n_c = d.groupby("c").size().values.astype(float)
    ka = (g[("a", "sum")] if hits_b is None else g["a"]).values
    kb = None if hits_b is None else g["b"].values
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(n_c), size=(B, len(n_c)))
    ra = ka[idx].sum(1) / n_c[idx].sum(1)
    point_a = ka.sum() / n_c.sum()
    out = {"n_rows": int(n_c.sum()), "n_clusters": int(len(n_c)), "B": B}
    if kb is None:
        out.update({"rate": float(point_a),
                    "ci95": [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))]})
        return out
    rb = kb[idx].sum(1) / n_c[idx].sum(1)
    point_b = kb.sum() / n_c.sum()
    if stat == "diff":
        s, p = rb - ra, point_b - point_a
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = rb / ra
        s = s[np.isfinite(s)]
        p = point_b / point_a if point_a else float("inf")
    out.update({"a": float(point_a), "b": float(point_b), stat: float(p),
                "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]})
    return out


# ---------------------------------------------------------------------------
# Yeniden uretilebilirlik: dosya hash'leri
# ---------------------------------------------------------------------------
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def model_paths(suffix=""):
    return {
        "rf": os.path.join(MODEL_DIR, f"MulticlassRF{suffix}.joblib"),
        "hgb": os.path.join(MODEL_DIR, f"MulticlassHGB{suffix}.joblib"),
        "lstm": os.path.join(MODEL_DIR, f"MulticlassLSTM{suffix}.keras"),
        "bundle": os.path.join(MODEL_DIR, f"MulticlassRF_bundle{suffix}.joblib"),
    }


def provenance(suffixes=("",), extra_files=()):
    """Kosuyu tanimlayan her girdinin hash'i + git HEAD."""
    files = {}
    for s in suffixes:
        for k, p in model_paths(s).items():
            if os.path.exists(p):
                files[os.path.relpath(p, PROJECT_DIR)] = sha256(p)
    for p in extra_files:
        if os.path.exists(p):
            files[os.path.relpath(p, PROJECT_DIR)] = sha256(p)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
                                       text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "multiclass_training", "analysis"],
            cwd=PROJECT_DIR, text=True).strip())
    except Exception:
        head, dirty = None, None
    return {"git_head": head, "git_dirty_code": dirty, "sha256": files}


def write_json(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    print(f"[+] {os.path.relpath(path, PROJECT_DIR)}")
    return path


def read_json(name):
    with open(os.path.join(OUT_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Korpus (CICIDS-2017 streaming dataset) -- egitimdeki batch-duzeyi test bolmesi
# ---------------------------------------------------------------------------
def load_corpus_split():
    """Egitimdeki `batch_level_split` ile BIREBIR ayni (train, test, classes).

    train_multiclass_models import edilir -- bolme mantigi kopyalanmaz."""
    import pandas as pd
    import contextlib
    import io
    import train_multiclass_models as tm

    df = pd.read_csv(DATA_CSV)
    with open(CLASSES_JSON, encoding="utf-8") as f:
        classes = json.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        tr, te = tm.batch_level_split(df, test_ratio=0.2, random_state=42)
    return tr, te, classes


def raw_columns():
    from pcap_validation_pilot_v5 import RAW_COLUMNS_V5
    return list(RAW_COLUMNS_V5)


def load_models(suffix=""):
    """(rf, hgb, lstm, pipeline, classes). Joblib RF/HGB (ONNX degil) -- korpus
    olcumleri icin; ONNX ile esdegerlik uretimde ayrica dogrulanmistir."""
    import joblib
    p = model_paths(suffix)
    bundle = joblib.load(p["bundle"])
    rf = joblib.load(p["rf"])
    hgb = joblib.load(p["hgb"]) if os.path.exists(p["hgb"]) else None
    lstm = None
    if os.path.exists(p["lstm"]):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        lstm = tf.keras.models.load_model(p["lstm"], compile=False)
    return rf, hgb, lstm, bundle["pipeline"], list(bundle["classes"])


def predict_all(models, X_raw_df):
    """Her model icin olasilik matrisi. X_raw_df: RAW_COLUMNS_V5 sutunlu DataFrame."""
    import numpy as np
    rf, hgb, lstm, pipeline, classes = models
    Xs = np.nan_to_num(pipeline.transform(X_raw_df), nan=0.0, posinf=0.0, neginf=0.0)
    out = {"rf": rf.predict_proba(Xs)}
    if hgb is not None:
        out["hgb"] = hgb.predict_proba(Xs)
    if lstm is not None:
        out["lstm"] = lstm.predict(Xs.reshape(-1, 1, Xs.shape[1]).astype("float32"),
                                   batch_size=8192, verbose=0)
    return out, Xs


def binary_metrics(proba, y_str, classes):
    """Ikili (argmax != BENIGN) recall/FPR + sinif bazinda recall, Wilson CI ile."""
    import numpy as np
    b = classes.index("BENIGN")
    pred = np.asarray(classes)[proba.argmax(1)]
    is_atk = (y_str != "BENIGN")
    alarm = (pred != "BENIGN")
    res = {
        "binary_recall": rate(int((alarm & is_atk).sum()), int(is_atk.sum())),
        "fpr": rate(int((alarm & ~is_atk).sum()), int((~is_atk).sum())),
        "per_class_recall": {},
    }
    for c in classes:
        if c == "BENIGN":
            continue
        m = (y_str == c)
        if m.sum():
            res["per_class_recall"][c] = rate(int(alarm[m].sum()), int(m.sum()))
    return res


# ---------------------------------------------------------------------------
# Gercek-dunya kollarini ayri surecte kos (offline_test_multiclass modelleri
# import aninda yukler -> sonek basina ayri surec zorunlu)
# ---------------------------------------------------------------------------
def run_realworld_arm(arm_name, suffix="", overrides=(), sets=None, extra_env=None,
                      reuse=True, dump=False):
    """realworld.py'yi alt-surecte calistirir, JSON sonucunu dondurur.

    overrides: [("Init_Win_bytes_backward", 28960.0), ...]  -- yalniz saldirgan
               satirlarina ve SYN-gorulmus akislara uygulanir.
    reuse: ayni (suffix, overrides) ile daha once kosulmus kol varsa onu dondur
           (kol ~2.5 dk; gate'ler arasinda ortak kollar -- orn. A0 == T1).
    """
    overrides = [(f, float(v)) for f, v in overrides]
    out_name = f"_rw_{arm_name}.json"
    cache = os.path.join(OUT_DIR, out_name)
    if reuse and os.path.exists(cache):
        prev = read_json(out_name)
        dumps_ok = (not dump) or all(os.path.exists(os.path.join(
            OUT_DIR, "packets", f"{arm_name}__{s['name']}.csv.gz")) for s in prev["sets"])
        if (prev.get("suffix") == suffix and dumps_ok and "provenance" in prev
                and [tuple(x) for x in prev.get("overrides", [])] == overrides
                and (sets is None or [s["name"] for s in prev["sets"]] == list(sets))):
            print(f"[arm {arm_name}] onbellekten ({out_name})", flush=True)
            return prev
    cmd = [sys.executable, os.path.join(THIS_DIR, "realworld.py"),
           "--arm", arm_name, "--out", out_name]
    for feat, val in overrides:
        cmd += ["--override", f"{feat}={val}"]
    if sets:
        cmd += ["--sets", ",".join(sets)]
    if dump:
        cmd += ["--dump"]
    env = dict(os.environ)
    env["MC_EVAL_SUFFIX"] = suffix
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if extra_env:
        env.update(extra_env)
    print(f"[arm {arm_name}] suffix='{suffix}' overrides={list(overrides)}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=PROJECT_DIR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"realworld arm {arm_name} basarisiz (rc={r.returncode})")
    return read_json(out_name)
