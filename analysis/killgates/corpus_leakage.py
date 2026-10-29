# -*- coding: utf-8 -*-
"""
Korpus-ici degerlendirmenin BITISIK-BATCH sizintisi: "yakin" vs "uzak" held-out.

Hakem itirazi (ND H1-M1, DG M4): ND Tablo 1'in "cross-day" held-out batch'leri
egitim batch'leriyle AYNI oturumdan ve aralarina serpistirilmis (orn. Thursday
WebAttack egitim 284-348, test 300-370); GATE-O3 WebAttack test satirlarinin
buyuk kisminin bir egitim satirina <0.01 mesafede oldugunu olctu. Soru: Tablo 1
rakamlari bu yakinliktan ne kadar sisiyor?

Protokol (SIFIR EGITIM -- uretim modelleri, uretim hatti `run_offline`):
  YAKIN : ND Tablo 1'in kendi held-out batch'leri (validate_all_days.GROUPS;
          DoS icin Wednesday dogrulama gruplari).
  UZAK  : ayni gun/ayni saldiri oturumu, ayni GT fonksiyonu; her EGITIM batch'ine
          indeks mesafesi >= MIN_GAP olan, >= MIN_ATTACK saldiri paketi tasiyan
          batch'ler (en hacimli MAX_FAR tanesi). Secim kurali asagida sabittir.
          Faz sinirlari: WebAttack < 380 (XSS/SQLi fazi Tablo 4'te ayrica
          olculdu), PortScan < 800 (820-848 port-80 DDoS trafigi -- ciftli GT
          ayirt edemez).
  Her batch icin:
    * tam hat: paket-duzeyi TP/FP/TN/FN (Tablo 1 ile ayni kural:
      alert = is_alert and not is_ood)
    * model: RF/HGB/LSTM argmax == aile (saldiri satirlari)
    * near-duplicate: her saldiri satirinin, ayni sinifin EGITIM satirlarina
      (uretim scaler uzayi) en yakin mesafesi; < 0.01 orani
    * zaman mesafesi: dosya adindaki zaman damgasina gore en yakin egitim batch'i
  Gruplar: havuzlanmis MCC/DR/FAR + BATCH-KUMELI bootstrap araligi.

Kontrol noktasi: her batch out/leakage/<gun>_<idx>.json'a yazilir; yeniden
kosumda atlanir.

Kullanim: python analysis/killgates/corpus_leakage.py
Cikti   : analysis/killgates/out/corpus_leakage.json
"""

import contextlib
import io
import json
import math
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pcap_validation_pilot as v1  # noqa: E402
import validate_all_days as vad  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import offline_test_multiclass as mc  # noqa: E402

MIN_GAP = 10
MIN_ATTACK = 200
MAX_FAR = 3
PHASE_BOUNDS = {"WebAttack": (0, 380), "PortScan": (0, 800)}
CKPT = os.path.join(common.OUT_DIR, "leakage")
DAY_OF = {"FTP-Patator": "Tuesday", "SSH-Patator": "Tuesday", "WebAttack": "Thursday",
          "PortScan": "Friday", "DDoS": "Friday", "DoS": "Wednesday"}
DOS_SCAN = range(600, 1030, 10)


def gt_for(family):
    ports = {"FTP-Patator": {21}, "SSH-Patator": {22}, "WebAttack": {80},
             "PortScan": None, "DDoS": {80}, "DoS": {80}}[family]
    return vad.make_pair_port_gt("172.16.0.1", "192.168.10.50", ports, family)


def batch_time(day, idx):
    p = vad.find_batch_day(day, idx)
    m = re.search(r"_(\d{14})\.pcap$", os.path.basename(p or ""))
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def near_batches():
    out = {}
    for day, fam, batches, _ in vad.GROUPS + vad.WEDNESDAY_GROUPS:
        f = "DoS" if fam.startswith("DoS") else fam
        if f in DAY_OF:
            out.setdefault(f, []).extend(batches)
    return out


def far_batches(train_idx, near):
    scan = json.load(open(os.path.join(common.PROJECT_DIR, "results",
                                       "_scan_train_candidates.json"), encoding="utf-8"))
    out = {}
    for fam, day in DAY_OF.items():
        tr = [int(b) for b in train_idx[fam]]
        if fam == "DoS":
            cands = []
            for i in DOS_SCAN:
                idx = f"{i:05d}"
                if min(abs(i - t) for t in tr) < MIN_GAP or idx in near[fam]:
                    continue
                pk = v1.read_pcap(v1.find_batch(idx)) if v1.find_batch(idx) else []
                g = gt_for(fam)
                cands.append({"idx": idx, "attack": sum(g(p)[0] for p in pk)})
        else:
            cands = scan[f"{day}/{fam}"]
        lo, hi = PHASE_BOUNDS.get(fam, (0, 10**9))
        ok = [c for c in cands
              if c["attack"] >= MIN_ATTACK and lo <= int(c["idx"]) < hi
              and c["idx"] not in near[fam]
              and min(abs(int(c["idx"]) - t) for t in tr) >= MIN_GAP]
        out[fam] = [c["idx"] for c in sorted(ok, key=lambda c: -c["attack"])[:MAX_FAR]]
    return out


class Recorder(mc.UniversalRealTimeFlowTableV5):
    last = None

    def __init__(self):
        super().__init__()
        self.rec = {}
        Recorder.last = self

    def extract_universal_features(self, packet):
        mf, fk = super().extract_universal_features(packet)
        if mf is not None:
            self.rec[str(packet.get("frame.number"))] = [float(mf.get(c, 0.0))
                                                         for c in mc.RAW_COLUMNS_V5]
        return mf, fk


def score_batch(fam, idx, models, extra, nn_by_class, train_times):
    path = os.path.join(CKPT, f"{DAY_OF[fam]}_{idx}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    rf, hgb, lstm_raw = extra
    day = DAY_OF[fam]
    pcap = v1.find_batch(idx) if fam == "DoS" else vad.find_batch_day(day, idx)
    packets = v1.read_pcap(pcap)
    gt = gt_for(fam)
    gt_by = {str(p.get("frame.number")): gt(p)[0] for p in packets}
    mc.UniversalRealTimeFlowTableV5 = Recorder
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = mc.run_offline(*models, packets)
    finally:
        mc.UniversalRealTimeFlowTableV5 = Recorder.__mro__[1]
    df = pd.DataFrame(res)
    df["k"] = df["packet_no"].astype(str)
    df = df[df["k"].isin(gt_by)]
    y = df["k"].map(gt_by).astype(int).values
    a = (df["is_alert"].astype(bool) & ~df["is_ood"].astype(bool)).values.astype(int)
    tp, fp = int(((a == 1) & (y == 1)).sum()), int(((a == 1) & (y == 0)).sum())
    tn, fn = int(((a == 0) & (y == 0)).sum()), int(((a == 0) & (y == 1)).sum())

    feats = pd.DataFrame.from_dict(Recorder.last.rec, orient="index", columns=mc.RAW_COLUMNS_V5)
    atk_keys = df.loc[y == 1, "k"]
    atk_keys = atk_keys[atk_keys.isin(feats.index)]
    X = np.nan_to_num(models[2].transform(feats.loc[atk_keys]))
    classes = models[3]
    fi = classes.index(fam)
    model_hits = {}
    if len(X):
        model_hits = {
            "rf": int((rf.predict_proba(X).argmax(1) == fi).sum()),
            "hgb": int((hgb.predict_proba(X).argmax(1) == fi).sum()),
            "lstm": int((lstm_raw.predict(X.reshape(-1, 1, 41).astype("float32"),
                                          batch_size=8192, verbose=0).argmax(1) == fi).sum())}
        d, _ = nn_by_class[fam].kneighbors(X)
        d = d[:, 0]
    else:
        d = np.array([])
    t = batch_time(day, idx)
    gap_s = min(abs((t - tt).total_seconds()) for tt in train_times[fam]) if t else None
    out = {"family": fam, "batch": idx, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "attack_rows_model": int(len(X)), "model_class_hits": model_hits,
           "nn_dist_p50": float(np.median(d)) if len(d) else None,
           "near_dup_lt_0.01": int((d < 0.01).sum()),
           "near_dup_lt_0.05": int((d < 0.05).sum()),
           "index_gap": int(min(abs(int(idx) - int(b)) for b in train_times[fam + "_idx"])),
           "time_gap_s": gap_s}
    os.makedirs(CKPT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def mcc(tp, fp, tn, fn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def pooled(rows, B=2000):
    s = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "tn", "fn")}
    rng = np.random.RandomState(common.SEED)
    arr = np.array([[r["tp"], r["fp"], r["tn"], r["fn"]] for r in rows], dtype=float)
    boots = {"mcc": [], "dr": []}
    for _ in range(B):
        t = arr[rng.randint(0, len(arr), len(arr))].sum(0)
        boots["mcc"].append(mcc(*t))
        boots["dr"].append(t[0] / max(t[0] + t[3], 1))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    n_atk = sum(r["attack_rows_model"] for r in rows)
    return {
        "batches": [r["batch"] for r in rows], **s,
        "mcc": mcc(s["tp"], s["fp"], s["tn"], s["fn"]), "mcc_ci95_batch_boot": ci(boots["mcc"]),
        "dr": s["tp"] / max(s["tp"] + s["fn"], 1), "dr_ci95_batch_boot": ci(boots["dr"]),
        "far": s["fp"] / max(s["fp"] + s["tn"], 1),
        "per_batch_mcc": {r["batch"]: round(mcc(r["tp"], r["fp"], r["tn"], r["fn"]), 4)
                          for r in rows},
        "model_class_recall": {m: sum(r["model_class_hits"].get(m, 0) for r in rows) / max(n_atk, 1)
                               for m in ("rf", "hgb", "lstm")},
        "near_dup_lt_0.01": sum(r["near_dup_lt_0.01"] for r in rows) / max(n_atk, 1),
        "near_dup_lt_0.05": sum(r["near_dup_lt_0.05"] for r in rows) / max(n_atk, 1),
        "index_gap": [r["index_gap"] for r in rows],
        "time_gap_s": [r["time_gap_s"] for r in rows],
    }


def main():
    from sklearn.neighbors import NearestNeighbors
    import joblib
    tr, _, _ = common.load_corpus_split()
    full = pd.read_csv(common.DATA_CSV, usecols=["family", "batch"])
    train_idx = {f: sorted(full.loc[full["family"] == f, "batch"].astype(int).unique())
                 for f in DAY_OF}
    near = near_batches()
    far = far_batches({f: [f"{b:05d}" for b in v] for f, v in train_idx.items()}, near)
    print("yakin:", near, "\nuzak:", far, flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        m = mc.load_multiclass_models()
    models = (m[0], m[1], m[2], list(m[3]), m[4], m[5])
    hgb = joblib.load(common.model_paths("")["hgb"])
    extra = (m[0], hgb, m[1])
    cols = common.raw_columns()
    nn_by_class = {}
    for f in DAY_OF:
        Xtr = np.nan_to_num(m[2].transform(tr.loc[tr["label_str"] == f, cols]))
        nn_by_class[f] = NearestNeighbors(n_neighbors=1).fit(Xtr)
    train_times = {}
    for f in DAY_OF:
        train_times[f + "_idx"] = [f"{b:05d}" for b in train_idx[f]]
        train_times[f] = [t for t in (batch_time(DAY_OF[f], b)
                                      for b in train_times[f + "_idx"]) if t]

    result = {"rule": {"min_index_gap": MIN_GAP, "min_attack_pkts": MIN_ATTACK,
                       "max_far": MAX_FAR, "phase_bounds": PHASE_BOUNDS},
              "families": {}}
    for fam in DAY_OF:
        r = {}
        for tag, batches in (("near_table1", near[fam]), ("far", far[fam])):
            rows = []
            for idx in batches:
                rows.append(score_batch(fam, idx, models, extra, nn_by_class, train_times))
                b = rows[-1]
                print(f"  {fam} {tag} {idx}: MCC={mcc(b['tp'], b['fp'], b['tn'], b['fn']):.4f} "
                      f"atk={b['tp'] + b['fn']} nd01={b['near_dup_lt_0.01']}/"
                      f"{b['attack_rows_model']} gap={b['index_gap']}", flush=True)
            r[tag] = pooled(rows) if rows else None
        result["families"][fam] = r
    result["provenance"] = common.provenance(("",), [common.DATA_CSV])
    common.write_json("corpus_leakage.json", result)
    for fam, r in result["families"].items():
        n, f = r["near_table1"], r["far"]
        print(f"{fam:12s} YAKIN MCC {n['mcc']:.4f} DR {n['dr']:.4f} nd {n['near_dup_lt_0.01']:.3f} | "
              + (f"UZAK MCC {f['mcc']:.4f} DR {f['dr']:.4f} nd {f['near_dup_lt_0.01']:.3f}"
                 if f else "UZAK yok"))


if __name__ == "__main__":
    main()
