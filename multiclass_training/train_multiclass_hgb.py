# -*- coding: utf-8 -*-
"""
train_multiclass_hgb.py
========================
R26 (davranissal-dal konsey vetosu, Faz 1): HistGradientBoostingClassifier'i
train_multiclass_models.py ile BIREBIR AYNI split/scaler/etiket uzayinda
egitir. Ayri bir bundle GEREKMEZ -- MulticlassRF_bundle.joblib'daki (pipeline,
classes, raw_columns, benign_index) HGB icin de gecerlidir (ayni egitim CSV'si,
ayni RAW_COLUMNS_V5, ayni scaler).

Amac: process_pcap_multiclass.py'nin davranissal (non-volumetrik) dalinda,
LSTM alarm verdiginde RF ve HGB'nin IKISI de BENIGN'e >=0.95 guvenle
katiliyorsa alarmi bastiran bir "konsey vetosu" (committee_dissent) icin
ikinci, LSTM'den mimari olarak BAGIMSIZ bir agac-tabanli sinif. RF zaten
hacimsel dalda yuklu; HGB bu betikle SIFIRDAN egitilir.

Hiperparametreler agent-memory arastirmasindaki (ids-generalization-strategist,
R26) olcumle AYNI (scratchpad/step3_minimal_committee_and_cost.py):
    HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                    random_state=13, class_weight="balanced")
O olcumde bu konfigurasyon: 7-sinif MCC=0.9674, WebAttack FP-veto (RF VE HGB
>=0.95) 839->108 (TP kaybi ~0). Baska bir hiperparametre TEKRAR ARANMADI --
zaten olculmus/dogrulanmis konfigurasyon aynen kullaniliyor.

Cikti: multiclass_training/models/MulticlassHGB.joblib
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, matthews_corrcoef

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from train_multiclass_models import batch_level_split, DATA_CSV, CLASSES_JSON

sys.path.insert(0, os.path.join(THIS_DIR, "..", "analysis"))
from pcap_validation_pilot_v5 import RAW_COLUMNS_V5

OUT_DIR = os.path.join(THIS_DIR, "models")
OUT_HGB = os.path.join(OUT_DIR, "MulticlassHGB.joblib")
BUNDLE_PATH = os.path.join(OUT_DIR, "MulticlassRF_bundle.joblib")


def main():
    print("[*] Dataset yukleniyor...", flush=True)
    df = pd.read_csv(DATA_CSV)
    with open(CLASSES_JSON, encoding="utf-8") as f:
        classes = json.load(f)

    print("[*] Batch-level train/test bolme (train_multiclass_models ile AYNI fonksiyon+seed)...", flush=True)
    df_train, df_test = batch_level_split(df, test_ratio=0.2, random_state=42)

    # RF/LSTM'nin egitimde kullandigi AYNI fit-edilmis scaler'i tekrar kullan
    # (yeniden fit etmek yerine) -- sayisal olarak birebir ayni transform garantisi.
    bundle = joblib.load(BUNDLE_PATH)
    scaler = bundle["pipeline"].named_steps["scaler"]
    assert bundle["raw_columns"] == RAW_COLUMNS_V5
    assert bundle["classes"] == classes

    label_to_int = {c: i for i, c in enumerate(classes)}
    X_train = scaler.transform(df_train[RAW_COLUMNS_V5].values.astype("float64"))
    y_train = df_train["label_str"].map(label_to_int).values.astype(int)
    X_test = scaler.transform(df_test[RAW_COLUMNS_V5].values.astype("float64"))
    y_test = df_test["label_str"].map(label_to_int).values.astype(int)
    print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}", flush=True)

    print("\n[*] HistGradientBoostingClassifier egitiliyor "
          "(max_iter=200, learning_rate=0.1, random_state=13, class_weight=balanced)...", flush=True)
    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, random_state=13, class_weight="balanced",
    )
    hgb.fit(X_train, y_train)
    pred = hgb.predict(X_test)

    print("\n--- HGB Test Sonuclari ---")
    print(classification_report(y_test, pred, target_names=classes, digits=4))
    mcc = matthews_corrcoef(y_test, pred)
    print(f"HGB MCC: {mcc:.4f}")

    joblib.dump(hgb, OUT_HGB)
    print(f"\n[+] Kaydedildi: {OUT_HGB}")


if __name__ == "__main__":
    main()
