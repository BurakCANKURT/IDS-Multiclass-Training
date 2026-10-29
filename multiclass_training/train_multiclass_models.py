# -*- coding: utf-8 -*-
"""
train_multiclass_models.py
===========================
build_multiclass_dataset.py ciktisi uzerinde multiclass RF + LSTM egitir.

MC1 (2026-08-05): kapsam 9 -> 7 sinif (Bot ve Infiltration cikarildi, bkz.
build_multiclass_streaming_dataset.py) + batch_level_split saldiri/benign
katmanlamasi duzeltildi (DoS test n=0, PortScan test n=10 yapaylig i giderildi).

Overfit onlemleri:
  RF  : min_samples_leaf=10, max_depth=25, max_features=0.5
  SMOTE: az ornekli siniflar icin sentetik oversample (guvenlik agi olarak
          birakildi). MC1 kapsaminda kalan 7 sinifin train ornek sayilari
          zaten hedeflerin uzerinde oldugundan SMOTE fiilen DEVRE DISI kalir
          -- yani uretim modeli sentetik ornek ICERMEZ. (Sentetik uretimden
          gercekten yararlanan tek sinif Bot idi; o da artik kapsam disi.)
  LSTM: iki Dropout katmani + L2 duzenlilik

Cikti (multiclass_training/models/):
  MulticlassRF.joblib
  MulticlassRF_bundle.joblib
  MulticlassLSTM.keras
"""

import os
import sys
import json
import random as _py_random
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# (R42, 2026-08-15) DETERMINIZM / TOHUM SABITLEME -- OLCUM ALTYAPISI DUZELTMESI
# ---------------------------------------------------------------------------
# GEREKCE: bu betikte RF (random_state=42) ve SMOTE (random_state=42) tohumlu,
# ama LSTM tohumsuzdu -> her egitim kosusu ~+-.03 MCC gurultusu tasiyordu
# ([[r38-xss-retrain-candidate-verdict]] Kanit D, [[external-dataset-acquisition
# -triage]] Bulgu 5.3). Bir veri-kompozisyonu deneyinin etkisi bu gurultunun
# altinda kalirsa OLCUM YORUMLANAMAZ. Bu blok tum non-determinizm kaynaklarini
# (Python hash/random, numpy global, TF/Keras agirlik-init + dropout +
# shuffle + op-level) tek bir tohuma baglar. `tf.keras.utils.set_random_seed`
# uygulama tarafina (main icinde, TF import'undan sonra) girer.
SEED = int(os.environ.get("MC_SEED", "42"))
os.environ["PYTHONHASHSEED"] = str(SEED)
_py_random.seed(SEED)

import numpy as np
import pandas as pd
import joblib

np.random.seed(SEED)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from pcap_validation_pilot_v5 import RAW_COLUMNS_V5

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
# (R42) VERSIYONLAMA / A-B: girdi CSV ve cikti model isimleri ortam
# degiskenleriyle degistirilebilir -> kontrol/deney kollari URETIM
# DOSYALARININ UZERINE YAZMAZ. Varsayilan (bos suffix) = mevcut uretim yolu;
# ama tum deney kosulari MC_MODEL_SUFFIX ile ayri isim kullanmalidir.
DATA_CSV     = os.environ.get(
    "MC_DATA_CSV", os.path.join(THIS_DIR, "data", "multiclass_streaming_dataset.csv"))
CLASSES_JSON = os.environ.get(
    "MC_CLASSES_JSON", os.path.join(THIS_DIR, "data", "streaming_label_classes.json"))
OUT_DIR      = os.path.join(THIS_DIR, "models")
_SUFFIX      = os.environ.get("MC_MODEL_SUFFIX", "")   # orn ".seedtest_a", ".r42_xss"
OUT_RF       = os.path.join(OUT_DIR, f"MulticlassRF{_SUFFIX}.joblib")
OUT_BUNDLE   = os.path.join(OUT_DIR, f"MulticlassRF_bundle{_SUFFIX}.joblib")
OUT_LSTM     = os.path.join(OUT_DIR, f"MulticlassLSTM{_SUFFIX}.keras")

N_FEAT = 41

# SMOTE hedef ornekleri — yeterli ornegi olan siniflara dokunmaz.
# MC1: Bot/Infiltration kapsam disi. Kalan siniflarin train sayilari
# (WebAttack 5124, FTP-Patator 8444, SSH-Patator 14527) hedefin uzerinde
# oldugu icin bu sozluk pratikte no-op'tur; sinif dengesizligi RF ve LSTM'de
# class_weight="balanced" ile ele alinir.
SMOTE_TARGETS = {
    "WebAttack":    5_000,
    "FTP-Patator":  5_000,
    "SSH-Patator":  5_000,
}


# MC1: bir batch'in "saldiri-temsili" sayilmasi icin gereken minimum saldiri
# orani. EXTRA_LOW_RATIO ile bilerek eklenen benign-arka-plan batch'leri
# (Friday 00758/00764 -> saldiri orani %0.08 ve %0.10) SIFIR degil ama
# ihmal edilebilir saldiri icerir; bunlari "saldiri batch'i" saymak, katmanli
# bolmenin PortScan test kumesini yine 10 ornege dusurmesine yol aciyordu.
# %1 esigi bu iki batch'i benign katmanina, gercek saldiri batch'lerinin
# tamamini (en dusugu Wednesday 00620 -> %4.4) saldiri katmanina koyar.
MIN_ATTACK_RATIO = 0.01


def _pick_test_batches(batches, test_ratio, temporal, random_state):
    """Bir batch kumesinden test'e ayrilacaklari secer (en az 1 batch train'de kalir)."""
    b = np.array(batches)
    if len(b) == 0:
        return set()
    if len(b) == 1:
        # Tek batch bolunemez -> train'de birak (test'e koyarsak train'de hic kalmaz).
        return set()
    if not temporal:
        rng = np.random.RandomState(random_state)
        rng.shuffle(b)
    n_test = max(1, round(len(b) * test_ratio))
    n_test = min(n_test, len(b) - 1)          # en az 1 batch train'de kalsin
    return set(b[-n_test:])                    # temporal: en gec batch'ler test'e


def batch_level_split(df, test_ratio: float = 0.2, random_state: int = 42,
                       temporal: bool = True):
    """
    Her pcap batch'ini ya tamamen train'e ya da tamamen test'e atar.
    Ayni batch'ten gelen satirlar her iki kumede birden olmaz (data leakage yok).

    temporal=True (varsayilan):
        Batch'leri indeks numarasina gore siralar (kucuk = erken zaman) ve
        son test_ratio kadarini test'e ayirir. Intra-session leakage'i azaltir:
        model erken trafikle egitilir, gec trafikle test edilir.

    MC1 DUZELTMESI — saldiri-tasiyan / saf-benign batch'ler AYRI katmanlanir
    ------------------------------------------------------------------------
    ESKI DAVRANIS (hatali): aile icindeki TUM batch'ler tek bir indeks
    siralamasina sokuluyordu. Ama iki aile icin en yuksek indeksli batch'ler
    kasitli olarak saldiri-fakiri secilmisti:
      * DoS      : TRAIN_BATCHES 17 adet SAF BENIGN Wednesday batch'i iceriyor
                   (20,60,...,1020,1150,...,1350). Saldiri penceresi 620-1000
                   arasinda; en gec 6 batch (1020-1350) %100 benign. Sonuc:
                   test'te DoS ornegi = 0  -> precision/recall/F1 = 0/0/0.
      * PortScan : EXTRA_LOW_RATIO ile bilerek eklenen dusuk-oranli 00758/00764
                   batch'leri (attack=8 ve 10) en yuksek indeksliler. Sonuc:
                   test'te PortScan ornegi = 10 -> metrikler anlamsiz.
    Yani sorun "veri kitligi" degil, temporal kuralin kasitli benign-agirlikli
    kuyruk batch'leriyle carpismasiydi (80.844 DoS ve 39.095 PortScan egitim
    ornegi mevcuttu ama hicbiri olcumlenemiyordu).

    YENI DAVRANIS: her aile icinde batch'ler once
        (a) saldiri-tasiyan  (>=1 BENIGN-disi satir)
        (b) saf-benign       (0 saldiri satiri)
    olarak ikiye ayrilir, temporal "son %20" kurali HER KATMANA AYRI uygulanir.
    Boylece her saldiri sinifi test'te gercek temsile kavusur; batch-duzeyi
    ayriklik (satir-duzeyi leakage yok) ve katman-ici zaman siralamasi korunur.
    """
    train_idx, test_idx = [], []

    for family, grp in df.groupby("family"):
        attack_ratio_per_batch = grp.groupby("batch")["label_str"].apply(
            lambda s: float((s != "BENIGN").sum()) / len(s)
        )
        rich = attack_ratio_per_batch >= MIN_ATTACK_RATIO
        attack_batches = np.sort(attack_ratio_per_batch[rich].index.values)
        benign_batches = np.sort(attack_ratio_per_batch[~rich].index.values)

        test_batches = (
            _pick_test_batches(attack_batches, test_ratio, temporal, random_state)
            | _pick_test_batches(benign_batches, test_ratio, temporal, random_state)
        )
        print(f"    [split] {family:<28} saldiri-batch={len(attack_batches):>2} "
              f"benign-batch={len(benign_batches):>2} -> test batch: {sorted(test_batches)}")

        is_test = grp["batch"].isin(test_batches)
        test_idx.extend(grp[is_test].index.tolist())
        train_idx.extend(grp[~is_test].index.tolist())

    return df.loc[train_idx], df.loc[test_idx]


def apply_smote(X_train, y_train, classes, targets: dict) -> tuple:
    """Her sinif icin ayri SMOTE uygular, zaten hedefte olan siniflara dokunmaz."""
    label_to_int = {c: i for i, c in enumerate(classes)}

    strategy = {}
    for cls_name, target in targets.items():
        idx = label_to_int.get(cls_name)
        if idx is None:
            continue
        current = int((y_train == idx).sum())
        if current < target:
            strategy[idx] = target
            print(f"    SMOTE: {cls_name} {current} -> {target}")
        else:
            print(f"    SMOTE: {cls_name} ({current}) zaten >= {target}, atlandi")

    if not strategy:
        return X_train, y_train

    # Cok az ornekli siniflar icin k_neighbors dusurulur (MC1 kapsaminda
    # bu dala hic girilmiyor; guvenlik agi olarak birakildi)
    min_class_size = min(int((y_train == idx).sum()) for idx in strategy)
    k = min(5, min_class_size - 1)
    k = max(k, 1)
    print(f"    SMOTE k_neighbors={k}")

    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"    SMOTE sonrasi: {len(y_train):,} -> {len(y_res):,} ornek")
    return X_res, y_res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Veri yukle ---
    print("[*] Dataset yukleniyor...", flush=True)
    df = pd.read_csv(DATA_CSV)
    with open(CLASSES_JSON, encoding="utf-8") as f:
        classes = json.load(f)
    N_CLASSES = len(classes)

    print(f"    {len(df):,} satir | {N_CLASSES} sinif: {classes}", flush=True)

    # --- Batch-level split (data leakage yok) ---
    # Ayni pcap batch'inden gelen satirlar train+test'e ayni anda dusmez.
    # Her aile icin batch'lerin ~%20'si test'e ayrilir.
    print("[*] Batch-level train/test bolme yapiliyor...", flush=True)
    df_train, df_test = batch_level_split(df, test_ratio=0.2, random_state=42)

    # Sinif dagilimini yazdir
    for split_name, split_df in [("Train", df_train), ("Test", df_test)]:
        vc = split_df["label_str"].value_counts()
        print(f"  {split_name} ({len(split_df):,} satir): "
              + " | ".join(f"{c}={vc.get(c,0)}" for c in classes))

    X_train = df_train[RAW_COLUMNS_V5].values.astype("float64")
    y_train  = df_train["label"].values.astype(int)
    X_test   = df_test[RAW_COLUMNS_V5].values.astype("float64")
    y_test   = df_test["label"].values.astype(int)
    print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}", flush=True)

    # --- Olcekleme (sadece train uzerinde fit) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- SMOTE (scaled uzayda uygula) ---
    print("\n[*] SMOTE uygulanıyor...", flush=True)
    X_train_sm, y_train_sm = apply_smote(X_train_s, y_train, classes, SMOTE_TARGETS)

    # -----------------------------------------------------------------------
    # RandomForest — overfit karsi regulerizasyon
    # -----------------------------------------------------------------------
    print("\n[*] RandomForest egitiliyor...", flush=True)
    print("    Parametreler: n_estimators=150, min_samples_leaf=10, "
          "max_depth=25, max_features=0.5", flush=True)
    rf = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=10,   # yaprak basina min 10 ornek — ezber azaltir
        max_depth=25,          # agac derinligi siniri
        max_features=0.5,      # her bolunmede ozelliklerin %50'si — agaclar arasi korelasyon azalir
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train_sm, y_train_sm)
    rf_pred = rf.predict(X_test_s)

    print("\n--- RF Test Sonuclari ---")
    print(classification_report(y_test, rf_pred, target_names=classes, digits=4))
    rf_mcc = matthews_corrcoef(y_test, rf_pred)
    print(f"RF MCC: {rf_mcc:.4f}")

    # 3-fold CV (overfit kontrolu — train vs CV farki onemli)
    print("\n[*] RF 3-fold cross-validation (train seti, SMOTE sonrasi)...", flush=True)
    cv_scores = cross_val_score(rf, X_train_sm, y_train_sm, cv=3,
                                scoring="matthews_corrcoef", n_jobs=-1)
    print(f"    CV MCC: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f}) | "
          f"Test MCC: {rf_mcc:.4f}")
    if cv_scores.mean() - rf_mcc > 0.05:
        print("    [!] CV-Test farki > 0.05 — hafif overfit sinyali var")
    else:
        print("    [+] CV-Test farki kabul edilebilir seviyede")

    # -----------------------------------------------------------------------
    # LSTM — iki Dropout + L2 duzenlilik
    # -----------------------------------------------------------------------
    print("\n[*] LSTM egitiliyor...", flush=True)
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras import regularizers

    # (R42) TF/Keras determinizmi: agirlik-init, dropout maskesi, fit shuffle'i
    # ve op-level indirgeme sirasini tohuma baglar. set_random_seed python+
    # numpy+tf'i birlikte ayarlar; enable_op_determinism CPU/GPU indirgemelerini
    # deterministik yapar (CPU-only bu makinede ihmal edilebilir yavaslama).
    # Bu cagri model olusturmadan ve fit'ten ONCE olmalidir.
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as _e:
        print(f"    [!] enable_op_determinism uygulanamadi ({_e}); "
              f"set_random_seed ile devam.", flush=True)
    print(f"    [+] Determinizm aktif (SEED={SEED})", flush=True)

    X_train_l = X_train_sm.reshape(-1, 1, N_FEAT).astype("float32")
    X_test_l  = X_test_s.reshape(-1, 1, N_FEAT).astype("float32")

    cw_arr = compute_class_weight(
        "balanced", classes=np.arange(N_CLASSES), y=y_train_sm
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(cw_arr)}
    print("    LSTM class_weight:", class_weight_dict, flush=True)

    lstm = Sequential([
        Input(shape=(1, N_FEAT)),
        LSTM(64, kernel_regularizer=regularizers.l2(1e-4),
                 recurrent_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(N_CLASSES, activation="softmax"),
    ])
    lstm.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    lstm.summary()

    lstm.fit(
        X_train_l, y_train_sm,
        validation_data=(X_test_l, y_test),
        epochs=20,
        batch_size=1024,
        class_weight=class_weight_dict,
        verbose=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        ],
    )

    lstm_pred = np.argmax(lstm.predict(X_test_l, verbose=0), axis=1)
    print("\n--- LSTM Test Sonuclari ---")
    print(classification_report(y_test, lstm_pred, target_names=classes, digits=4))
    print(f"LSTM MCC: {matthews_corrcoef(y_test, lstm_pred):.4f}")

    # -----------------------------------------------------------------------
    # Kaydet
    # -----------------------------------------------------------------------
    joblib.dump(rf, OUT_RF)

    bundle = {
        "raw_columns":  RAW_COLUMNS_V5,
        "pipeline":     Pipeline(steps=[("scaler", scaler)]),
        "classes":      classes,
        "n_classes":    N_CLASSES,
        "label_to_int": {c: i for i, c in enumerate(classes)},
        "int_to_label": {i: c for i, c in enumerate(classes)},
        "benign_index": classes.index("BENIGN"),
    }
    joblib.dump(bundle, OUT_BUNDLE)
    lstm.save(OUT_LSTM)

    print(f"\n[+] Kaydedildi:")
    print(f"    {OUT_RF}")
    print(f"    {OUT_BUNDLE}")
    print(f"    {OUT_LSTM}")


if __name__ == "__main__":
    main()
