# -*- coding: utf-8 -*-
"""
Welford / canli-offline ozellik paritesi -- KALICI TEST.

2026-08-21'de process_pcap_multiclass.py'deki sinirsiz-buyuyen fwd_lengths /
bwd_lengths listeleri O(1) artimli sayaclara (running sum/max/min + Welford)
cevrildi. O gun esdegerlik scratchpad'de 4000 rastgele senaryoyla dogrulanmisti;
bu dosya o dogrulamayi kalici hale getirir ve bir adim ileri gider:

  1. test_welford_matches_numpy
       _welford_update, 4000 rastgele dizide sum/len ve np.var(ddof=0) ile esdeger.
  2. test_live_matches_offline_on_real_pcap
       Canli hattin akis tablosu (Welford) ile offline referans tablosu
       (analysis/pcap_validation_pilot_v5, liste tabanli) AYNI gercek pcap
       uzerinde paket paket ayni 41 ozelligi uretir.

process_pcap_multiclass.py import aninda canli dongusunu baslattigi icin
dogrudan import EDILMEZ; yalnizca ust-duzey fonksiyon/sinif tanimlari ve
sabit atamalari AST ile yalitilip yuklenir.

Kullanim:
    python -m pytest tests/test_welford_flow_parity.py -q
    python tests/test_welford_flow_parity.py          # pytest'siz
"""

import ast
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
# The live detector lives in the separate live-console repository.
# Point IDS_LIVE_DETECTOR at its process_pcap_multiclass.py to run the parity
# tests; without it they are skipped.
LIVE_SRC = os.environ.get(
    "IDS_LIVE_DETECTOR",
    os.path.join(ROOT, "multiclass_training", "process_pcap_multiclass.py"))
# Any real capture with >1000 IP packets works; override with IDS_PARITY_PCAP.
PCAP = os.environ.get(
    "IDS_PARITY_PCAP",
    os.path.join(ROOT, "data", "realworld", "capture_esp32_ftp_softvote_20260814_1657.pcapng"))
N_SCENARIOS = 4000
# std/var Welford ile iki-gecisli np.var arasinda yuvarlama farki tasir;
# geri kalan ozellikler bit-birebir olmalidir.
FLOAT_TOL_FEATURES = {"Bwd Packet Length Std", "Packet Length Std",
                      "Packet Length Variance"}


def load_live_namespace():
    if not os.path.exists(LIVE_SRC):
        import pytest
        pytest.skip(f"live detector not found (set IDS_LIVE_DETECTOR): {LIVE_SRC}")
    tree = ast.parse(open(LIVE_SRC, encoding="utf-8").read())
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            keep.append(n)
        elif isinstance(n, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id.isupper() for t in n.targets):
            try:
                ast.literal_eval(n.value)
                keep.append(n)
            except Exception:
                pass
    ns = {"np": np, "math": math, "re": re, "os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), LIVE_SRC, "exec"), ns)
    return ns


def test_welford_matches_numpy():
    upd = load_live_namespace()["_welford_update"]
    rng = np.random.RandomState(20260821)
    worst_mean = worst_var = 0.0
    for i in range(N_SCENARIOS):
        n = int(rng.choice([1, 2, 3, rng.randint(4, 50), rng.randint(50, 2000)]))
        kind = i % 4
        if kind == 0:
            xs = rng.randint(40, 1515, n).astype(float)           # paket boyu
        elif kind == 1:
            xs = np.full(n, float(rng.randint(40, 1515)))         # sabit
        elif kind == 2:
            xs = rng.choice([60.0, 1514.0], n)                    # iki tepe
        else:
            xs = rng.randint(40, 65535, n).astype(float)          # LSO super-cerceve
        cnt, mean, m2 = 0, 0.0, 0.0
        for x in xs:
            cnt, mean, m2 = upd(cnt, mean, m2, float(x))
        assert cnt == n
        worst_mean = max(worst_mean, abs(mean - xs.mean()) / max(abs(xs.mean()), 1.0))
        worst_var = max(worst_var, abs(m2 / n - xs.var()) / max(xs.var(), 1.0))
    assert worst_mean < 1e-12, worst_mean
    assert worst_var < 1e-9, worst_var


def test_live_matches_offline_on_real_pcap():
    if not os.path.exists(PCAP):
        import pytest
        pytest.skip(f"pcap yok: {PCAP}")
    import pcap_validation_pilot as v1
    from pcap_validation_pilot_v5 import UniversalRealTimeFlowTableV5, RAW_COLUMNS_V5
    live = load_live_namespace()["UniversalRealTimeFlowTable"]()
    ref = UniversalRealTimeFlowTableV5()
    n = 0
    worst = {}
    exact_mismatch = []
    for p in v1.read_pcap(PCAP):
        a, _ = live.extract_universal_features(p)
        b, _ = ref.extract_universal_features(p)
        assert (a is None) == (b is None)
        if a is None:
            continue
        n += 1
        for c in RAW_COLUMNS_V5:
            x, y = float(a.get(c, 0.0)), float(b.get(c, 0.0))
            if c in FLOAT_TOL_FEATURES:
                worst[c] = max(worst.get(c, 0.0), abs(x - y) / max(abs(y), 1.0))
            elif x != y and len(exact_mismatch) < 10:
                exact_mismatch.append((n, c, x, y))
    assert n > 1000, n
    assert not exact_mismatch, exact_mismatch
    assert all(v < 1e-9 for v in worst.values()), worst


if __name__ == "__main__":
    test_welford_matches_numpy()
    print(f"[OK] Welford == numpy ({N_SCENARIOS} senaryo)")
    test_live_matches_offline_on_real_pcap()
    print("[OK] canli (Welford) == offline (liste) ozellikleri, gercek pcap")
