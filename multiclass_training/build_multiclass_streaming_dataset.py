# -*- coding: utf-8 -*-
"""
build_multiclass_streaming_dataset.py
======================================
build_streaming_dataset_v5.py ile AYNI pcap batch kompozisyonunu kullanir;
tek fark: binary label (0/1) yerine multiclass label_str uretir.

  binary_label == 0  ->  label_str = "BENIGN"
  binary_label == 1  ->  label_str = family  ("DoS", "DDoS", ...)

Bu sayede LSTM, test sirasinda kullanilan streaming pcap ozellikleri
(UniversalRealTimeFlowTableV5) ile ayni dagilimdaki verilerle egitilir
-- ISCX CSV (CICFlowMeter batch) kaynakli dagilim farki ortadan kalkar.

Cikti:
  multiclass_training/data/multiclass_streaming_dataset.csv
  multiclass_training/data/streaming_label_classes.json

MC1 (2026-08-05) — KAPSAM DARALTMA: Bot + Infiltration siniftan cikarildi.
  Bot          : RF precision 0.123 / LSTM 0.177 (test n=154) — bir SOC icin
                 her gercek alarma karsilik 5-7 yanlis alarm demek; savunulamaz.
  Infiltration : metrikler mukemmel gorunuyor ama yalnizca ~36 HAM ornege
                 dayaniyor (SMOTE ile 300'e sisirilmis) — uretim icin
                 istatistiksel olarak fazla ince.
Hedef kapsam 7 sinif: BENIGN, DDoS, DoS, FTP-Patator, PortScan, SSH-Patator,
WebAttack.

Bu iki ailenin SALDIRI satirlari atilir; ayni batch'lerin BENIGN satirlari
KORUNUR (bkz. KEEP_BENIGN_FROM_EXCLUDED) — aksi halde ~101K gercek benign
ornek (Friday bot-saati + Thursday infiltration-saati arka plan trafigi)
kaybedilir ve kalan 7 sinifin yanlis-pozitif orani bosuna kotulesir.

Calistirma:
  python build_multiclass_streaming_dataset.py                 # pcap'lerden bastan uret (yavas)
  python build_multiclass_streaming_dataset.py --from-existing # mevcut CSV'yi filtreleyerek uret (hizli)
Iki mod da AYNI satir kumesini uretir: cikarim per-batch deterministik oldugu
icin filtreleme, yeniden isleme ile birebir esdegerdir.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Proje analysis modulleri
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "analysis"))

import pcap_validation_pilot as v1
from pcap_validation_pilot_v5 import UniversalRealTimeFlowTableV5, RAW_COLUMNS_V5
from validate_all_days import find_batch_day
from _scan_train_candidates import SCAN_DEFS
from build_streaming_dataset import TRAIN_BATCHES

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_JSON = os.path.join(PROJECT_DIR, "results", "_scan_train_candidates.json")
OUT_CSV   = os.path.join(THIS_DIR, "data", "multiclass_streaming_dataset.csv")
OUT_JSON  = os.path.join(THIS_DIR, "data", "streaming_label_classes.json")

# build_streaming_dataset_v5.py ile ayni secim parametreleri; TOP_K=6 daha fazla cesitlilik
TOP_K = 6
EXTRA_LOW_RATIO = {
    "Friday/PortScan": ["00758", "00764"],
    "Friday/DDoS":     ["00758", "00764"],
}
INFILTRATION_EXTRA = ["00697", "00679", "00682"]

# --- MC1 kapsam filtresi ---------------------------------------------------
# Bu ailelerin SALDIRI satirlari (label_str == aile adi) veri setinden atilir.
EXCLUDED_ATTACK_FAMILIES = {"Bot", "Infiltration"}
# True: cikarilan ailelerin batch'lerindeki BENIGN satirlari korunur (benign
# cesitliligi kaybolmasin diye). family kolonu asagidaki adla yeniden yazilir
# ki train/test bolme bunlari "saldiri iceren aile" sanmasin.
KEEP_BENIGN_FROM_EXCLUDED = True
EXCLUDED_BENIGN_FAMILY_NAME = {
    "Bot":          "Benign-Friday-BotWindow",
    "Infiltration": "Benign-Thursday-InfilWindow",
}


def apply_scope_filter(df):
    """MC1: kapsam disi birakilan ailelerin saldiri satirlarini dusurur."""
    before = len(df)
    drop_mask = df["label_str"].isin(EXCLUDED_ATTACK_FAMILIES)
    n_drop = int(drop_mask.sum())
    df = df[~drop_mask].copy()

    if KEEP_BENIGN_FROM_EXCLUDED:
        for fam, new_name in EXCLUDED_BENIGN_FAMILY_NAME.items():
            df.loc[df["family"] == fam, "family"] = new_name
    else:
        df = df[~df["family"].isin(EXCLUDED_ATTACK_FAMILIES)].copy()

    print(f"[MC1] kapsam filtresi: {before:,} -> {len(df):,} satir "
          f"({n_drop:,} saldiri satiri dusuruldu: {sorted(EXCLUDED_ATTACK_FAMILIES)})")
    if KEEP_BENIGN_FROM_EXCLUDED:
        kept = int(df["family"].isin(EXCLUDED_BENIGN_FAMILY_NAME.values()).sum())
        print(f"[MC1] cikarilan ailelerin BENIGN satirlari korundu: {kept:,}")
    return df.reset_index(drop=True)


def _extract_multiclass(pcap, gt_func, family, day, idx):
    """Paketleri isle: binary gt -> multiclass label_str."""
    packets = v1.read_pcap(pcap)
    flow_table = UniversalRealTimeFlowTableV5()
    rows = []
    n_attack = 0
    for pkt in packets:
        master_features, _ = flow_table.extract_universal_features(pkt)
        if master_features is None:
            continue
        binary_label, _ = gt_func(pkt)
        label_str = "BENIGN" if binary_label == 0 else family
        n_attack += binary_label
        row = {col: float(master_features.get(col, 0.0)) for col in RAW_COLUMNS_V5}
        row["label_str"] = label_str
        row["day"]       = day
        row["family"]    = family
        row["batch"]     = idx
        rows.append(row)
    print(f"    -> {len(packets)} pkt | {n_attack} attack ({family})", flush=True)
    return rows


def main(from_existing: bool = False):
    os.makedirs(os.path.join(THIS_DIR, "data"), exist_ok=True)

    if from_existing:
        # Hizli yol: pcap'leri yeniden islemeden mevcut CSV'yi filtrele.
        # Cikarim per-batch deterministik oldugundan sonuc tam yol ile ayni.
        print(f"[*] Mevcut CSV filtreleniyor: {OUT_CSV}")
        df_old = pd.read_csv(OUT_CSV)
        df = apply_scope_filter(df_old)
        _finalize(df)
        return

    rows = []

    # --- 1) Wednesday 28-batch (DoS: Slowloris/Slowhttptest/Hulk) ---
    print(f"\n{'='*60}")
    print(f"=== Wednesday/DoS ({len(TRAIN_BATCHES)} batch) ===")
    for idx in TRAIN_BATCHES:
        pcap = v1.find_batch(idx)
        if not pcap:
            print(f"[!] batch {idx} bulunamadi, atlaniyor.")
            continue
        print(f"  [batch {idx}] {os.path.basename(pcap)}", flush=True)
        rows += _extract_multiclass(pcap, v1.gt_label_with_info if hasattr(v1, 'gt_label_with_info')
                                    else lambda pkt: (v1.gt_label(pkt), "DoS"),
                                    "DoS", "Wednesday", idx)

    # --- 2) Tue/Thu/Fri (SCAN_DEFS TOP_K=3 + EXTRA_LOW_RATIO) ---
    gt_funcs = {f"{day}/{family}": gt_func for day, family, gt_func, *_ in SCAN_DEFS}
    with open(SCAN_JSON, "r", encoding="utf-8") as f:
        scan = json.load(f)

    for key, candidates in scan.items():
        day, family = key.split("/")
        gt_func = gt_funcs.get(key)
        if gt_func is None:
            print(f"[!] {key} icin gt_func bulunamadi, atlaniyor.")
            continue
        by_idx = {c["idx"]: c for c in candidates}
        extra  = EXTRA_LOW_RATIO.get(key, [])
        k      = max(TOP_K - len(extra), 1)
        picked = candidates[:k] + [by_idx[i] for i in extra if i in by_idx]
        print(f"\n=== {key}: {len(picked)} batch ===", flush=True)
        for c in picked:
            idx  = c["idx"]
            pcap = find_batch_day(day, idx)
            print(f"  [{idx}] {os.path.basename(pcap)} (attack={c['attack']}/{c['total']})", flush=True)
            rows += _extract_multiclass(pcap, gt_func, family, day, idx)

    # --- 3) Infiltration extra 3 batch ---
    # TOP_K artinca bu batch'ler zaten yukaridaki dongude secilmis olabilir;
    # sadece henuz eklenmemis olanlari ekle (tekrar isleme / data duplikasyonu yok).
    inf_gt = gt_funcs.get("Thursday/Infiltration")
    if inf_gt:
        already_picked = {str(r.get("batch", "")) for r in rows
                          if r.get("family") == "Infiltration"}
        extra_new = [idx for idx in INFILTRATION_EXTRA if idx not in already_picked]
        if extra_new:
            print(f"\n=== Thursday/Infiltration extra ({len(extra_new)} yeni batch) ===", flush=True)
            for idx in extra_new:
                pcap = find_batch_day("Thursday", idx)
                print(f"  [{idx}] {os.path.basename(pcap)}", flush=True)
                rows += _extract_multiclass(pcap, inf_gt, "Infiltration", "Thursday", idx)
        else:
            print(f"\n=== Thursday/Infiltration extra: TOP_K={TOP_K} ile zaten kapsandi, atlandi ===", flush=True)

    # --- Dataframe ve label encode ---
    df = pd.DataFrame(rows, columns=RAW_COLUMNS_V5 + ["label_str", "day", "family", "batch"])
    df = apply_scope_filter(df)
    _finalize(df)


def _finalize(df):
    """Ortak son adim: label encode + dagilim raporu + diske yazma."""
    classes      = sorted(df["label_str"].unique())
    label_to_int = {c: i for i, c in enumerate(classes)}
    df["label"]  = df["label_str"].map(label_to_int)

    print(f"\n{'='*60}")
    print(f"Toplam satir : {len(df):,}")
    print(f"Sinif sayisi : {len(classes)}")
    print("Dagilim:")
    for cls in classes:
        cnt = (df["label_str"] == cls).sum()
        print(f"  [{label_to_int[cls]}] {cls:<20} {cnt:>8,}")

    df[RAW_COLUMNS_V5 + ["label", "label_str", "day", "family", "batch"]].to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Dataset : {OUT_CSV}")
    print(f"[+] Siniflar: {OUT_JSON}")


if __name__ == "__main__":
    main(from_existing="--from-existing" in sys.argv)
