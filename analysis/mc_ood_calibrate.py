# -*- coding: utf-8 -*-
"""Multiclass OOD-gate calibration -- same method as R20 (binary), new data.

Reference manifold: multiclass_training/data/multiclass_streaming_dataset.csv,
label_str==BENIGN, non-volumetric-eligible mask (matches production routing:
rate<500k | rate==1e6 (single-packet-flow artifact, R10 class) | DNS bucket).
Scaled with the MULTICLASS model's OWN pipeline (MulticlassRF_bundle.joblib) --
NOT the binary one, different fit data.

Then measures 1-NN Euclidean distance (same numpy-matvec method as R20, see
pipeline/process_pcap.py::ood_distance) for:
  - a disjoint BENIGN holdout sample from the same CSV
  - each attack class in the CSV (in-training, same caveat R20's "egitim ATK"
    rows had)
  - a genuinely held-out population: Wednesday Slowloris/Slowhttptest pcap
    batches NOT used to build multiclass_streaming_dataset.csv (same 10
    batches R20 used for the binary gate), features extracted with the
    v5-parity flow table (pcap_validation_pilot_v5), scaled with the
    multiclass pipeline.

Usage:
    # from the repository root:
    python analysis/mc_ood_calibrate.py
Output:
    multiclass_training/models/MulticlassRF__ood_reference.npz  (ref, sqnorm)
    stdout distance-report table (used to pick OOD_DISTANCE_THRESHOLD)
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "analysis"))
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pcap_validation_pilot as v1  # noqa: E402
import pcap_validation_pilot_v5 as v5mod  # noqa: E402

RNG = np.random.default_rng(20260806)

CSV_PATH = os.path.join(PROJECT_DIR, "multiclass_training", "data", "multiclass_streaming_dataset.csv")
BUNDLE_PATH = os.path.join(PROJECT_DIR, "multiclass_training", "models", "MulticlassRF_bundle.joblib")
OUT_REF_PATH = os.path.join(PROJECT_DIR, "multiclass_training", "models", "MulticlassRF__ood_reference.npz")

VOLUMETRIC_THRESHOLD = 500_000
N_REF = 20_000
N_HOLDOUT = 20_000
N_PER_ATTACK = 10_000

HELDOUT_BATCHES = ["00595", "00600", "00610", "00630", "00635", "00640",
                    "00645", "00655", "00665", "00670"]


def load_bundle():
    b = joblib.load(BUNDLE_PATH)
    return b["pipeline"], b["raw_columns"], b["classes"], b["benign_index"]


def nonvol_mask(df):
    rate = df["Flow Packets/s"]
    return (rate < VOLUMETRIC_THRESHOLD) | (rate == 1_000_000) | (df["PortBucket_dns"] == 1)


def scale(pipeline, raw_columns, df):
    X = df[raw_columns].astype(float).values
    return pipeline.transform(X)


def build_reference(pipeline, raw_columns, df):
    mask = nonvol_mask(df) & (df["label_str"] == "BENIGN")
    pool = df[mask]
    print(f"[*] BENIGN non-volumetric-eligible pool: {len(pool)} / {len(df)} total rows")
    idx = RNG.permutation(len(pool))
    ref_idx = idx[:N_REF]
    holdout_idx = idx[N_REF:N_REF + N_HOLDOUT]
    ref_df = pool.iloc[ref_idx]
    holdout_df = pool.iloc[holdout_idx]
    print(f"[*] reference={len(ref_df)}  disjoint_holdout={len(holdout_df)}")

    ref_scaled = scale(pipeline, raw_columns, ref_df).astype(np.float32)
    sqnorm = (ref_scaled ** 2).sum(axis=1).astype(np.float32)
    return ref_scaled, sqnorm, holdout_df


def nn_distance_batch(ref_scaled, sqnorm, X_scaled):
    """1-NN Euclidean distance of each row in X_scaled to ref_scaled, via matvec."""
    X = np.asarray(X_scaled, dtype=np.float32)
    xn = (X ** 2).sum(axis=1)
    # d2[i,j] = sqnorm[j] - 2*X[i]@ref[j] + xn[i]
    cross = X @ ref_scaled.T   # (n, n_ref)
    d2 = sqnorm[None, :] - 2.0 * cross + xn[:, None]
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2.min(axis=1))


def report(name, dists):
    d = np.asarray(dists)
    if len(d) == 0:
        print(f"  {name:35s} n=0 (skipped)")
        return
    p50, p99, p999, mx = np.percentile(d, 50), np.percentile(d, 99), np.percentile(d, 99.9), d.max()
    n_gt30 = int((d > 30).sum())
    print(f"  {name:35s} n={len(d):6d}  p50={p50:6.3f}  p99={p99:6.3f}  p99.9={p999:6.3f}  max={mx:7.2f}  >30={n_gt30}")


def extract_heldout_features():
    """Extract raw v5 features from the held-out Wednesday Slowloris/Slowhttptest
    pcap batches using the v5-parity flow table (SAME feature extractor
    binary/multiclass both use). Filters to non-volumetric-eligible +
    fwd_packets>=1 (mirrors R20's methodology: excludes structural R17-flip
    OOD artifacts that R18 already suppresses, to isolate genuine temporal
    signal)."""
    all_rows = []
    for idx in HELDOUT_BATCHES:
        pcap = v1.find_batch(idx)
        if not pcap:
            print(f"[!] batch {idx} not found, skipping")
            continue
        packets = v1.read_pcap(pcap)
        ft = v5mod.UniversalRealTimeFlowTableV5()
        for pkt in packets:
            mf, fk = ft.extract_universal_features(pkt)
            # (R28) update_volumetric artik 8 deger donduruyor (is_conn_flood,
            # distinct_conns eklendi); bu kalibrasyon yalnizca is_flood/is_scan/
            # rate ile "non-volumetrik dal maskesi" kuruyor -- yeni sinyal de
            # ayni maskeye dahil edilmeli, aksi halde referans kumesine artik
            # RF dalina giden paketler sizar.
            (is_flood, vol_pkts, vol_syn, vol_ratio, is_scan, distinct_ports,
             is_conn_flood, distinct_conns) = ft.update_volumetric(pkt)
            if mf is None or fk is None:
                continue
            rate = mf.get("Flow Packets/s", 0.0)
            is_nonvol_eligible = (rate < VOLUMETRIC_THRESHOLD) or (rate == 1_000_000) or (mf.get("PortBucket_dns", 0.0) == 1.0)
            # (R28) yeni deterministik hacimsel tetikleyiciler de referansi
            # kirletmemeli (DNS muafiyeti disinda).
            if (is_flood or is_scan or is_conn_flood) and mf.get("PortBucket_dns", 0.0) != 1.0:
                is_nonvol_eligible = False
            flow = ft.flows.get(fk)
            fwd_pkts = flow.get("fwd_packets", 0) if flow else 0
            if is_nonvol_eligible and fwd_pkts >= 1:
                all_rows.append(mf)
        print(f"    batch {idx}: {len(packets)} pkts read, {len(all_rows)} cumulative eligible rows")
    return pd.DataFrame(all_rows)


def main():
    print("[*] Loading multiclass bundle...")
    pipeline, raw_columns, classes, benign_idx = load_bundle()
    print(f"    classes={classes}")

    print("[*] Loading multiclass streaming dataset CSV (this may take a bit)...")
    df = pd.read_csv(CSV_PATH)
    print(f"    {len(df)} rows loaded")

    ref_scaled, sqnorm, holdout_df = build_reference(pipeline, raw_columns, df)

    print("\n[*] Distance report (1-NN Euclidean, multiclass-scaled space)\n")

    holdout_scaled = scale(pipeline, raw_columns, holdout_df)
    d = nn_distance_batch(ref_scaled, sqnorm, holdout_scaled)
    report("BENIGN disjoint holdout", d)

    # (S5) Conformal p-deger kalibrasyonu: disjoint holdout, REFERANStan AYRI
    # bir BENIGN kume oldugu icin gecerli bir kalibrasyon kumesidir (referans
    # noktalarinin kendine mesafesi trivially 0 olurdu, kalibrasyon icin
    # KULLANILAMAZ). Sirali sakla -> inference'ta np.searchsorted ile O(log n)
    # p-deger hesaplanir. bkz. process_pcap_multiclass.py::ood_pvalue().
    calib_scores = np.sort(d.astype(np.float32))
    print(f"\n[*] Saving reference + conformal kalibrasyon -> {OUT_REF_PATH}")
    np.savez(OUT_REF_PATH, ref=ref_scaled, sqnorm=sqnorm, calib_scores=calib_scores)

    for cls in classes:
        if cls == "BENIGN":
            continue
        sub_mask = nonvol_mask(df) & (df["label_str"] == cls)
        sub = df[sub_mask]
        if len(sub) == 0:
            print(f"  {cls:35s} n=0 (no non-volumetric-eligible rows in CSV)")
            continue
        if len(sub) > N_PER_ATTACK:
            sub = sub.iloc[RNG.permutation(len(sub))[:N_PER_ATTACK]]
        sub_scaled = scale(pipeline, raw_columns, sub)
        d = nn_distance_batch(ref_scaled, sqnorm, sub_scaled)
        report(f"train-set ATK {cls} (non-vol rows)", d)

    print("\n[*] Extracting HELD-OUT Wednesday Slowloris/Slowhttptest pcap batches "
          f"({HELDOUT_BATCHES})...")
    heldout_df = extract_heldout_features()
    print(f"    {len(heldout_df)} eligible packets extracted")
    if len(heldout_df) > 0:
        heldout_scaled = scale(pipeline, raw_columns, heldout_df)
        d_heldout = nn_distance_batch(ref_scaled, sqnorm, heldout_scaled)
        report("HELD-OUT Slowloris/Slowhttptest (fwd>=1)", d_heldout)

    # (S5) mevcut OOD_DISTANCE_THRESHOLD=30.0'in kalibrasyon kumesindeki
    # ampirik p-degeri ne? (n_calib + 1'e bolen +1 duzeltmesiyle -- standart
    # conformal p-deger formulu, bkz. ood_pvalue() docstring'i)
    n_calib = len(calib_scores)
    exceed = int((calib_scores >= 30.0).sum())
    p_at_30 = (1 + exceed) / (n_calib + 1)
    print(f"\n[*] Conformal dogrulama: esik=30.0 -> kalibrasyon kumesinde "
          f"(n={n_calib}) {exceed} nokta >= 30.0 -> ampirik p-deger <= {p_at_30:.6f} "
          f"(yani en fazla ~%{p_at_30*100:.4f} gercek-benign trafik bu esigi asar)")

    print("\nDone.")


if __name__ == "__main__":
    main()
