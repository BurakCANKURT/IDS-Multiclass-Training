# -*- coding: utf-8 -*-
"""
offline_test_multiclass.py
===========================
offline_test.ipynb'nin multiclass uyarlamasi.
process_pcap.py ile AYNI dispatcher mimarisini kullanir;
tek fark: MainRF_v5/MainLSTM_v5 (ikili) yerine MulticlassRF/MulticlassLSTM
(MC1 kapsami, 7 sinif: BENIGN, DDoS, DoS, FTP-Patator, PortScan,
SSH-Patator, WebAttack) kullanilmasi ve risk_score + saldiri adi buna gore
uretilmesidir. Bot ve Infiltration MC1'de kapsam disi birakildi
(Bot: precision 0.12-0.18; Infiltration: ~36 ham ornek).

Calistirilma:
  # from the repository root:
  python multiclass_training/offline_test_multiclass.py

Kullanim:
  CSV_PATH   : canli_trafik_kaydi.csv yolu
  MAX_PACKETS: islenecek maksimum paket sayisi (None = tumu)
"""

import csv
import io
import json
import os
import sys
import warnings
import zipfile
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import h5py
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Proje yollarini ayarla ────────────────────────────────────────────────
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "analysis"))

from pcap_validation_pilot_v5 import (
    UniversalRealTimeFlowTableV5,
    RAW_COLUMNS_V5,
    _port_bucket,
)

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
CSV_PATH    = os.path.join(PROJECT_DIR, "data", "canli_trafik_kaydi.csv")
# (R42 parite) model dosya yollari ortam degiskeniyle degistirilebilir --
# versiyonlu aday (orn .r42_xss) modelleri URETIM DOSYALARINA DOKUNMADAN
# degerlendirmek icin. Varsayilan = uretim. MC_DISABLE_ONNX=1, RF/HGB ONNX'i
# devre disi birakip joblib'e dusurur (aday ONNX'i yoksa mismatch olmasin diye;
# konsey vetosu da no-op olur -- A/B'nin IKI kolunda da esit uygulanir).
_MSUF       = os.environ.get("MC_EVAL_SUFFIX", "")   # orn ".r42_xss" / ".seedtest_a"
MC_BUNDLE   = os.environ.get("MC_EVAL_BUNDLE", os.path.join(THIS_DIR, "models", f"MulticlassRF_bundle{_MSUF}.joblib"))
MC_RF       = os.environ.get("MC_EVAL_RF",     os.path.join(THIS_DIR, "models", f"MulticlassRF{_MSUF}.joblib"))
MC_LSTM     = os.environ.get("MC_EVAL_LSTM",   os.path.join(THIS_DIR, "models", f"MulticlassLSTM{_MSUF}.keras"))
MC_DISABLE_ONNX = os.environ.get("MC_DISABLE_ONNX", "0") == "1"

MAX_PACKETS = 10_000   # None = tum paketler
PRINT_EACH  = False    # Her paketi yazdir? (cok olunca False yap)

# ── Dispatcher esikleri (offline_test.ipynb ile AYNI; ayrintili gerekce
#    icin process_pcap.py - binary pipeline - uzerindeki yorumlara bakiniz) ──
SYN_BURST_MIN      = 100   # bir kaynaktan tek hedefe minimum SYN sayisi (flood imzasi)
SYN_RATIO_MIN      = 0.5   # akistaki SYN-only paket orani esigi
PORT_SCAN_MIN_PORTS = 40   # bu sayida farkli dst_port -> is_scan=True
PORT_SCAN_FULL_PORTS = 80  # bu sayida farkli portta scan_conf = 1.0
# (R28/O1.1-O1.3 parite -- gerekce process_pcap.py'de). NOT: is_flood/
# is_conn_flood SAYAC mantigi pcap_validation_pilot.py::update_volumetric'ten
# MIRAS alinir (UniversalRealTimeFlowTableV5); burada yalnizca guven
# formullerinin sabitleri gerekiyor.
SYN_ABS_MIN = 150
SYN_ABS_FULL = 300
CONN_FLOOD_MIN = 200
CONN_FLOOD_FULL = 400
SCAN_RF_VETO_MAX_PORTS = 150  # (R28/O1.2) bu port sayisinin ALTINDA RF vetosu gecerli
ALERT_THRESHOLD    = 0.4   # risk_score bu degeri gecince ATTACK sayilir
LSTM_WEIGHT        = 0.6   # blended_score agirligi: LSTM
COUNTER_WEIGHT     = 0.4   # blended_score agirligi: Progressive Counter
VOLUMETRIC_FEATURE = "Flow Packets/s"
VOLUMETRIC_THRESHOLD = 500_000  # R10 kok-neden duzeltmesi: tek-paketlik-akis rate yapaylığını asar
VOLUMETRIC_WINDOW_SECONDS = 60  # (R27 parite) process_pcap_multiclass.py VOL_WINDOW_SECONDS ile ayni
# (R33 parite) akis-basi SABIT-PAYDALI paket hizi penceresi -- gerekce ve esik
# kalibrasyonu process_pcap.py'deki R33 yorumunda. PILOT_R33=0 ile R33-oncesi
# davranis geri gelir (A/B icin; pcap_validation_pilot.py ile ayni env degiskeni).
R33_RATE_WINDOW = os.environ.get("PILOT_R33", "1") != "0"
RATE_WINDOW_SECONDS = 0.1
RATE_TRIGGER_PPS = 25_000.0
RATE_WIN_MIN_PKTS = int(RATE_TRIGGER_PPS * RATE_WINDOW_SECONDS)  # 2500 pkt/100ms
PROG_MIN_PKTS      = 12    # (R27/O3 parite) canli hatla AYNI (eskiden 10 idi, LSTM_FLOOR_PER_CLASS'in
                            # canlida kullandigi deger buydu -- bkz. process_pcap_multiclass.py PROG_MIN_PKTS)
EARLY_ATTACK_LSTM  = 0.85  # erken-saldiri istisnasi icin LSTM skor esigi
# (R34 parite) 1'de KALIR; muafiyet artik kosulsuz degil --
# committee_waives_early_attack() guvencesi eklendi. Gerekce/olcum ve
# reddedilen MIN_SUSP=2 varyanti: analysis/pcap_validation_pilot.py'deki
# ayni adli sabitin yorum blogunda (kanonik).
EARLY_ATTACK_MIN_SUSP = 1
INIT_WIN_CLIENT_MAX = 512

# (R19/R20 parite -- bkz. process_pcap_multiclass.py) OOD/kapsam-disi kapisi.
# Ayni referans dosyasi (multiclass'in KENDI PIPELINE'iyla kalibre edildi,
# bkz. o dosyadaki yorum + agent-memory/.../project_r22_volumetric_scan_latch_gap.md).
# Bu betikte risk_score/is_alert HESABINA DOKUNULMAZ (Bulgu-3 ile ayni durum:
# proje-genelinde hicbir dogrulama harness'i OOD'yu DR/FAR'a katmiyor) --
# yalnizca is_ood/ood_distance ek sutun olarak raporlanir.
OOD_REF_PATH = os.environ.get(
    "MC_EVAL_OOD_REF",
    os.path.join(THIS_DIR, "models", f"MulticlassRF__ood_reference{_MSUF}.npz"))
OOD_DISTANCE_THRESHOLD = 30.0
OOD_REF = None
OOD_SQNORM = None
OOD_CALIB_SCORES = None  # (S5 parite) conformal p-deger icin
if os.path.exists(OOD_REF_PATH):
    try:
        _ood_z = np.load(OOD_REF_PATH)
        OOD_REF = _ood_z["ref"].astype(np.float32)
        OOD_SQNORM = _ood_z["sqnorm"].astype(np.float32)
        if "calib_scores" in _ood_z:
            OOD_CALIB_SCORES = _ood_z["calib_scores"].astype(np.float32)
        print(f"[*] OOD kapisi aktif: {os.path.basename(OOD_REF_PATH)} "
              f"({OOD_REF.shape[0]} referans nokta, esik={OOD_DISTANCE_THRESHOLD}, "
              f"conformal-kalibrasyon={'var' if OOD_CALIB_SCORES is not None else 'yok'})")
    except Exception as _e:
        print(f"[!] OOD referansi yuklenemedi ({_e}); OOD kapisi DEVRE DISI.")
else:
    print(f"[!] OOD referans dosyasi bulunamadi ({OOD_REF_PATH}); OOD kapisi DEVRE DISI.")


def ood_distance(scaled_row):
    if OOD_REF is None:
        return None
    v = np.asarray(scaled_row, dtype=np.float32).ravel()
    d2 = OOD_SQNORM - 2.0 * (OOD_REF @ v) + float(v @ v)
    return float(np.sqrt(max(float(d2.min()), 0.0)))


def ood_pvalue(dist):
    """(S5 parite -- bkz. process_pcap_multiclass.py) risk_score/is_alert'e
    DOKUNMAZ, sadece raporlanir."""
    if OOD_CALIB_SCORES is None or dist is None:
        return None
    n = OOD_CALIB_SCORES.shape[0]
    idx = int(np.searchsorted(OOD_CALIB_SCORES, dist, side="left"))
    exceed = n - idx
    return float((1 + exceed) / (n + 1))


# (R26 Faz 1 + R27 parite -- bkz. process_pcap_multiclass.py) ONNX modelleri:
# (a) davranissal-dal konsey vetosu, (b) volumetrik dalin RF-ONNX'i (joblib
# fallback ile). RF/HGB BAGIMSIZ yuklenir, isimler bir kez cache'lenir (D1),
# calisma-zamani hatalari fail-safe yakalanir (O2 parite).
COMMITTEE_VETO_THRESHOLD = 0.95


def _load_onnx_session(path):
    if not os.path.exists(path):
        return None
    try:
        import onnxruntime as _ort
        so = _ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = _ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        return (sess, sess.get_inputs()[0].name, sess.get_outputs()[1].name)
    except Exception as _e:
        print(f"[!] ONNX yuklenemedi ({os.path.basename(path)}): {_e}")
        return None


_RF_ONNX = None if MC_DISABLE_ONNX else _load_onnx_session(os.path.join(THIS_DIR, "models", f"MulticlassRF{_MSUF}.onnx"))
_HGB_ONNX = None if MC_DISABLE_ONNX else _load_onnx_session(os.path.join(THIS_DIR, "models", f"MulticlassHGB{_MSUF}.onnx"))
print(f"[*] RF-ONNX: {'aktif' if _RF_ONNX else 'yok (joblib fallback)'} | "
      f"Konsey vetosu: {'aktif' if (_RF_ONNX and _HGB_ONNX) else 'DEVRE DISI'} "
      f"(esik={COMMITTEE_VETO_THRESHOLD})")


def _onnx_proba_row(onnx_tuple, scaled_row):
    sess, in_name, prob_name = onnx_tuple
    x = np.ascontiguousarray(np.asarray(scaled_row).reshape(1, -1), dtype=np.float32)
    out = sess.run([prob_name], {in_name: x})[0]
    return np.asarray(out, dtype=np.float64)[0]


def _rf_onnx_proba_or_none(scaled_row):
    """(R27/Y2 parite) Volumetrik dal icin; hata/eksiklik -> None, cagiran joblib'e duser."""
    if _RF_ONNX is None:
        return None
    try:
        return _onnx_proba_row(_RF_ONNX, scaled_row)
    except Exception as _e:
        print(f"[!] RF-ONNX calisma-zamani hatasi, joblib'e dusuluyor: {_e}")
        return None


def committee_dissents(scaled_row, benign_idx):
    """(R26 Faz 1 parite) RF VE HGB ikisi de BENIGN'e >= esik guvenle
    katiliyorsa True. Konsey devre disiysa (ONNX yok) VEYA calisma-zamani
    hatasinda (O2 parite) her zaman False."""
    if _RF_ONNX is None or _HGB_ONNX is None:
        return False
    try:
        p_rf = _onnx_proba_row(_RF_ONNX, scaled_row)
        p_hgb = _onnx_proba_row(_HGB_ONNX, scaled_row)
    except Exception as _e:
        print(f"[!] Konsey ONNX calisma-zamani hatasi, veto atlaniyor: {_e}")
        return False
    return p_rf[benign_idx] >= COMMITTEE_VETO_THRESHOLD and p_hgb[benign_idx] >= COMMITTEE_VETO_THRESHOLD


# (R34 parite) early_attack muafiyetini bloke etmek icin gereken minimum
# BENIGN olasiligi (0.5 = "argmax BENIGN" esdegeri). Gerekce/olcum:
# analysis/pcap_validation_pilot.py EARLY_ATTACK_MIN_SUSP yorum blogu.
COMMITTEE_WAIVER_BENIGN_P = 0.5


def committee_waives_early_attack(scaled_row, benign_idx):
    """(R34 parite) True = RF VE HGB ikisi de BENIGN sayiyor -> early_attack
    muafiyeti VERILMEZ. Fail-safe yon R26'nin TERSI (hata/eksik konsey ->
    False, muafiyet eski davranisla verilir)."""
    if _RF_ONNX is None or _HGB_ONNX is None:
        return False
    try:
        p_rf = _onnx_proba_row(_RF_ONNX, scaled_row)
        if p_rf[benign_idx] < COMMITTEE_WAIVER_BENIGN_P:
            return False
        p_hgb = _onnx_proba_row(_HGB_ONNX, scaled_row)
    except Exception as _e:
        print(f"[!] Konsey/muafiyet ONNX calisma-zamani hatasi, atlaniyor: {_e}")
        return False
    return p_hgb[benign_idx] >= COMMITTEE_WAIVER_BENIGN_P


# ── (R26 FAZ 2 parite) DAVRANISSAL DALDA 3-UYE SOFT-VOTE ───────────────────
# Kanonik gerekce/olcum: process_pcap_multiclass.py'deki MC_R26_SOFTVOTE
# yorum blogu. Ozet: Faz 1 (committee_dissents) yalnizca ATTACK->BENIGN
# yonunde calisan bir VETO'dur; HANGI saldiri sinifinin secildigine hic
# karismaz ve LSTM'in kendi karar yuzeyini degistirmez. Faz 2, davranissal
# dalin KARAR OLASILIK VEKTORUNU LSTM+RF+HGB'nin agirlikli ortalamasi yapar.
# VARSAYILAN KAPALI (env MC_R26_SOFTVOTE=1 ile acilir) -- mevcut uretim
# davranisi bit-duzeyinde korunur.
MC_R26_SOFTVOTE = os.environ.get("MC_R26_SOFTVOTE", "0") == "1"
_SV_W = os.environ.get("MC_R26_SOFTVOTE_W", "")
if _SV_W:
    SOFTVOTE_WEIGHTS = tuple(float(x) for x in _SV_W.split(","))
    assert len(SOFTVOTE_WEIGHTS) == 3, "MC_R26_SOFTVOTE_W = w_lstm,w_rf,w_hgb"
else:
    SOFTVOTE_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

SOFTVOTE_STATS = {"applied": 0, "fallback": 0}


def softvote_proba(lstm_proba, scaled_row):
    """(R26 Faz 2 parite) LSTM+RF+HGB olasilik vektorlerinin agirlikli
    ortalamasi. Kapali/ONNX yok/calisma-zamani hatasi -> None (cagiran SAF
    LSTM'e duser = mevcut uretim davranisi; R26 Faz 1 ile AYNI fail-safe
    felsefesi)."""
    if not MC_R26_SOFTVOTE or _RF_ONNX is None or _HGB_ONNX is None:
        return None
    try:
        p_rf = _onnx_proba_row(_RF_ONNX, scaled_row)
        p_hgb = _onnx_proba_row(_HGB_ONNX, scaled_row)
    except Exception as _e:
        print(f"[!] Soft-vote ONNX calisma-zamani hatasi, saf LSTM'e dusuluyor: {_e}")
        SOFTVOTE_STATS["fallback"] += 1
        return None
    w_l, w_r, w_h = SOFTVOTE_WEIGHTS
    SOFTVOTE_STATS["applied"] += 1
    return (w_l * np.asarray(lstm_proba, dtype=np.float64)
            + w_r * p_rf + w_h * p_hgb)


print(f"[*] R26 Faz 2 soft-vote: {'AKTIF' if MC_R26_SOFTVOTE else 'kapali (varsayilan)'} "
      f"agirlik(LSTM,RF,HGB)={SOFTVOTE_WEIGHTS}")


# ---------------------------------------------------------------------------
# Dinamik LSTM yukleyici (offline_test.ipynb Cell 4 ile ayni)
# ---------------------------------------------------------------------------
def build_and_load_dynamic_lstm(keras_path: str):
    print(f"[+] Multiclass LSTM yukleniyor: {os.path.basename(keras_path)}")
    extracted = {}
    with zipfile.ZipFile(keras_path, "r") as z:
        h5_files = [n for n in z.namelist() if n.endswith(".h5") or n.endswith(".weights.h5")]
        target = next((n for n in h5_files if "model.weights" in n or "model" in n), None)
        if target is None and h5_files:
            target = h5_files[0]
        if target:
            with z.open(target) as f_h5:
                buf = io.BytesIO(f_h5.read())
            with h5py.File(buf, "r") as hf:
                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        extracted[name] = np.array(obj)
                hf.visititems(visitor)

    kernel_keys = [k for k in extracted if "kernel" in k and ("lstm" in k.lower() or "0" in k)]
    dense_keys  = [k for k in extracted if "kernel" in k and "dense" in k]

    if not kernel_keys:
        print("    [!] Agirlik cikarilamadi, dogrudan tf.keras.models.load_model deneniyor")
        return tf.keras.models.load_model(keras_path)

    lstm_kernel = extracted[kernel_keys[0]]
    input_dim = lstm_kernel.shape[0]

    dense_last_key = sorted(dense_keys)[-1]
    n_classes = extracted[dense_last_key].shape[1]

    print(f"    input_dim={input_dim}, n_classes={n_classes}")

    model = tf.keras.models.load_model(keras_path)
    return model


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar (offline_test.ipynb Cell 8 ile ayni)
# ---------------------------------------------------------------------------
def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_tcp_flags(value):
    if value is None or value == "":
        return 0
    try:
        if isinstance(value, str):
            s = value.strip()
            return int(s, 16) if s.lower().startswith("0x") else int(s)
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_multicast_ip(ip_str):
    """(R23 parite -- bkz. process_pcap_multiclass.py) mDNS/SSDP gibi
    link-local kesif protokolleri multicast hedeflere gider ve TANIM GEREGI
    bir "kurban" hedefleyemez; multiclass LSTM bunlari orta-guvenle
    FTP-Patator/SSH-Patator'a benzetiyordu (OOD kapisi yakalamadi, mesafe
    egitim-benign mertebesinde). CICIDS'te multicast trafik OLMADIGI icin bu
    kontrol bu betikte pratikte NO-OP'tur, sadece parite icin eklendi."""
    if not ip_str:
        return False
    if ":" in ip_str:
        return ip_str.lower().startswith("ff")
    parts = ip_str.split(".")
    if len(parts) != 4:
        return False
    try:
        first = int(parts[0])
    except ValueError:
        return False
    return 224 <= first <= 239


def _is_broadcast_ip(ip_str, src_ip=None):
    """(R32 parite -- bkz. pipeline/process_pcap.py::_is_broadcast_ip) IPv4
    sinirli-yayin adresi 255.255.255.255; R23 multicast kontrolunun (224/4)
    kapsamadigi kardes vaka. CICIDS'te yayin trafigi OLMADIGI icin bu betikte
    NO-OP'tur, sadece canli/offline parite icin eklendi.
    (R52 parite -- bkz. process_pcap_multiclass.py) ALT-AG YONLENDIRILMIS
    YAYIN: dst_ip son okteti 255 ve src_ip ile ilk 3 oktet ayniysa da True."""
    if ip_str == "255.255.255.255":
        return True
    if src_ip:
        d = ip_str.split(".")
        s = src_ip.split(".")
        if len(d) == 4 and len(s) == 4 and d[3] == "255" and d[:3] == s[:3]:
            return True
    return False


# (R27/O3 parite -- bkz. process_pcap_multiclass.py _get_src_ip/_get_dst_ip/
# _get_proto/_is_non_ip) eskiden bu betik ham `packet.get("ip.src","0.0.0.0")`
# kullanip IPv6 fallback'i ATLIYORDU ve ARP/saf-L2 paketlerini flow-table'a
# SOKUYORDU -- canli hat R17'den beri bunlari filtreliyor/fallback'liyor.
def _get_src_ip(packet):
    v = packet.get("ip.src")
    if v is None or v == "":
        v = packet.get("ipv6.src")
    return v if v not in (None, "") else "0.0.0.0"


def _get_dst_ip(packet):
    v = packet.get("ip.dst")
    if v is None or v == "":
        v = packet.get("ipv6.dst")
    return v if v not in (None, "") else "0.0.0.0"


def _get_proto(packet, default=6):
    v = packet.get("ip.proto")
    if v in (None, ""):
        v = packet.get("ipv6.nxt")
    return _to_int(v, default)


def _is_non_ip(packet):
    return (
        packet.get("ip.src") in (None, "") and packet.get("ipv6.src") in (None, "")
        and packet.get("ip.dst") in (None, "") and packet.get("ipv6.dst") in (None, "")
    )


def _sanitize_matrix(arr):
    """(R27/O3 parite) canli hat scaled_input'u tahminden once NaN/Inf'ten
    temizliyor; bu betik hic temizlemiyordu."""
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)


# ---------------------------------------------------------------------------
# Modelleri yukle
# ---------------------------------------------------------------------------
def load_multiclass_models():
    print("[*] Multiclass modeller yukleniyor...")
    bundle = joblib.load(MC_BUNDLE)
    rf     = joblib.load(MC_RF)
    lstm   = build_and_load_dynamic_lstm(MC_LSTM)

    classes      = bundle["classes"]
    n_classes    = bundle["n_classes"]
    benign_idx   = bundle["benign_index"]
    pipeline     = bundle["pipeline"]
    raw_columns  = bundle["raw_columns"]

    assert raw_columns == RAW_COLUMNS_V5, "raw_columns uyusmazligi!"
    print(f"    {n_classes} sinif: {classes}")
    print(f"    BENIGN index: {benign_idx}")
    return rf, lstm, pipeline, classes, n_classes, benign_idx


# ---------------------------------------------------------------------------
# Canli trafik CSV'sini oku (offline_test.ipynb Cell 10 ile ayni)
# ---------------------------------------------------------------------------
def load_packets(csv_path: str, max_packets=None):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            rows.append(row)
    print(f"[+] {len(rows):,} paket yuklendi: {os.path.basename(csv_path)}")
    return rows


# ---------------------------------------------------------------------------
# Ana dispatcher dongusu
# ---------------------------------------------------------------------------
def _precompute_lstm(rows, pipeline, lstm, n_classes, benign_idx):
    """Geçiş-1: tüm paketler için features çıkar, tek seferde batch LSTM predict yap."""
    print("[*] Geçiş-1: tüm paket özellikleri çıkarılıyor (LSTM batch predict için)...")
    ft_pre = UniversalRealTimeFlowTableV5()
    scaled_list = []
    for packet in rows:
        mf, _ = ft_pre.extract_universal_features(packet)
        if mf is None:
            mf = {col: 0.0 for col in RAW_COLUMNS_V5}
        row_df = pd.DataFrame(
            [{col: float(mf.get(col, 0.0)) for col in RAW_COLUMNS_V5}],
            columns=RAW_COLUMNS_V5,
        )
        scaled_list.append(pipeline.transform(row_df))

    X_all = np.vstack(scaled_list).reshape(-1, 1, N_FEAT).astype("float32")
    print(f"[*] LSTM batch predict ({len(rows):,} paket)...")
    lstm_probas = lstm.predict(X_all, batch_size=1024, verbose=1)   # shape (N, n_classes)
    print(f"[+] Batch predict tamamlandi: {lstm_probas.shape}")
    return {i: lstm_probas[i] for i in range(len(rows))}


# ---------------------------------------------------------------------------
# Strateji sabitleri
# ---------------------------------------------------------------------------
# Strateji B (revize): Sinif ozgu olgunluk kapisi esigi
#   Skor dokundurulmaz; sadece hangi LSTM_ALERT_FLOOR kullanilacagi sinifa gore degisir.
#   DDoS neredeyse her zaman volumetrik dalla yakalanir; LSTM temporal dali icin cok
#   yuksek esik (0.98) CDN/web trafiklerini kapida tutar. Gercek DoS-Hulk ise
#   susp_pkts biriktirir -> early_attack=True -> kapi atlanan.
LSTM_FLOOR_PER_CLASS = {
    "DDoS":      0.98,   # CDN cok-kaynak trafiklerini bastirmak icin cok yuksek esik
    "DoS":       0.92,   # binary varsayilandan biraz yuksek
    # MC1: "Bot" girdisi kaldirildi (sinif kapsam disi).
    "WebAttack": 0.97,   # port-80 CDN overlap → yuksek floor ile FP kesme
}
DEFAULT_LSTM_FLOOR = 0.95   # diger siniflar

# ── (R26 Faz 2 parite) SOFT-VOTE ICIN YENIDEN KALIBRE EDILMIS FLOOR'LAR ────
# Kanonik gerekce/olcum: process_pcap_multiclass.py'deki SOFTVOTE_FLOOR_MAP
# yorum blogu. VARSAYILAN KAPALI; yalnizca MC_R26_SOFTVOTE=1 IKEN ve
# MC_R26_SOFTVOTE_FLOORS=1 verilirse devreye girer.
MC_R26_SOFTVOTE_FLOORS = os.environ.get("MC_R26_SOFTVOTE_FLOORS", "0") == "1"
# (R26 Faz 2 / EARLY-RAW parite) bkz. process_pcap_multiclass.py yorum blogu.
MC_R26_SOFTVOTE_EARLY_RAW = (os.environ.get("MC_R26_SOFTVOTE_EARLY_RAW", "0") == "1"
                            and MC_R26_SOFTVOTE)
SOFTVOTE_FLOOR_MAP = {"DDoS": 0.8669, "DoS": 0.7173, "WebAttack": 0.7867}
SOFTVOTE_DEFAULT_FLOOR = 0.7458
SOFTVOTE_EARLY_ATTACK_LSTM = 0.6669

if MC_R26_SOFTVOTE and MC_R26_SOFTVOTE_FLOORS:
    ACTIVE_FLOOR_MAP = SOFTVOTE_FLOOR_MAP
    ACTIVE_DEFAULT_FLOOR = SOFTVOTE_DEFAULT_FLOOR
    ACTIVE_EARLY_ATTACK_LSTM = SOFTVOTE_EARLY_ATTACK_LSTM
    print(f"[*] R26 Faz 2: floor'lar YENIDEN KALIBRE EDILMIS setle calisiyor "
          f"({SOFTVOTE_FLOOR_MAP}, default={SOFTVOTE_DEFAULT_FLOOR}, "
          f"early_attack={SOFTVOTE_EARLY_ATTACK_LSTM})")
else:
    ACTIVE_FLOOR_MAP = LSTM_FLOOR_PER_CLASS
    ACTIVE_DEFAULT_FLOOR = DEFAULT_LSTM_FLOOR
    ACTIVE_EARLY_ATTACK_LSTM = EARLY_ATTACK_LSTM

# Strateji A: CDN / cok-hedefe-konusan kaynak filtresi
#   Bir src_ip bu kadar farkli dst_ip'e konusmussa CDN/legitim sunucu sayilir.
CDN_DST_THRESHOLD = 3

# Strateji D: DoS/DDoS kaynak sayisi disambiguasyonu
#   Tek hedefe saldiran benzersiz src sayisi bu esikten azsa DoS; fazlaysa DDoS.
DDOS_MIN_UNIQUE_SRC = 3

# ── (R37/D1 parite) HACIMSEL DALDA PORT-FARKINDALIGI ───────────────────────
# Kanonik gerekce/olcum: process_pcap_multiclass.py'deki APP_LAYER_FLOOD_PORTS
# yorum blogu. Ozet: hacimsel dalin RF-override kolu (imzanin etiketi KENDISI
# urettigi durum) dst_port'u hic gormuyordu; port-80/3000 uygulama-katmani
# selleri kosulsuz "DoS" etiketleniyordu (R36, 1798 kayit). Bu betikte etki
# YALNIZCA `detected_attack_name` uzerindedir -- risk_score/is_alert'e
# DOKUNULMAZ, dolayisiyla DR/FAR/MCC uzerinde YAPISAL OLARAK etkisizdir
# (ONCE/SONRA birebir ayni olmasi BEKLENIR ve dogrulanir).
APP_LAYER_FLOOD_PORTS = {80, 443, 3000, 8000, 8080, 8443, 8888}
R37_D1_PORT_AWARE_FLOOD = os.environ.get("MC_R37_D1", "1") != "0"


def _flood_override_name(prefix, flood_label, dst_port):
    """(R37/D1 parite) bkz. process_pcap_multiclass.py::_flood_override_name."""
    port = _to_int(dst_port, 0)
    if not R37_D1_PORT_AWARE_FLOOD:
        return f"{prefix} -> {flood_label} (RF-override)"
    if port in APP_LAYER_FLOOD_PORTS:
        return (f"{prefix} -> {flood_label} (RF-override, port={port}: uygulama-katmanı "
                f"seli -- {flood_label}/WebAttack-flood ayrımı BELİRSİZ)")
    return f"{prefix} -> {flood_label} (RF-override, port={port})"


# ── (R37/D2 parite) STRATEJI-D KAYNAK KUMESINDE PAKET-YONU KARISIKLIGI ─────
# Kanonik gerekce: process_pcap_multiclass.py'deki R37_D2_ORIENTED_DDOS_SRC
# yorum blogu. Ozet: anahtar `(PAKETIN dst_ip'si, AKISIN dst_port'u)` seklinde
# yari-paket/yari-akis duzeyindeydi; host'un kendi giden trafigine gelen YANIT
# paketleri uzak sunuculari "saldirgan kaynak" olarak yaziyor ve tek-kaynakli
# DoS'u DDoS'a yukseltiyordu. Artik hem anahtar hem kaynak akisin KANONIK
# yonunden turetilir ve kume YALNIZCA ileri-yon paketlerde guncellenir.
R37_D2_ORIENTED_DDOS_SRC = os.environ.get("MC_R37_D2", "1") != "0"


def _ddos_src_context(flow_table_obj, flow_key, src_ip, dst_ip, dst_port):
    """(R37/D2 parite) bkz. process_pcap_multiclass.py::_ddos_src_context."""
    if not R37_D2_ORIENTED_DDOS_SRC:
        return (dst_ip, dst_port), src_ip, True
    flow = flow_table_obj.flows.get(flow_key) if flow_key is not None else None
    if flow is None:
        return (dst_ip, dst_port), src_ip, True
    fwd_src_key = flow.get("fwd_src_key") or ""
    client_ip = fwd_src_key.rsplit(":", 1)[0] if ":" in fwd_src_key else src_ip
    is_fwd = (client_ip == src_ip)
    target_ip = dst_ip if is_fwd else src_ip
    return (target_ip, dst_port), client_ip, is_fwd


def _apply_dos_ddos_disambiguate(pred_class, key, src_for_set, is_fwd,
                                 attack_srcs_per_target, now_s):
    """Strateji D: tahmin DDoS/DoS ise kaynak sayisina gore duzelt.
    (R27 parite -- bkz. process_pcap_multiclass.py) VOL_WINDOW_SECONDS kayan
    pencere: bu script'in kisa (10k paket) kosularinda pratikte etkisiz ama
    parite icin ayni desen kullanilir.
    (R37/D2 parite) anahtar/kaynak _ddos_src_context()'ten gelir; kume
    yalnizca ileri-yon paketlerde guncellenir."""
    if pred_class not in ("DDoS", "DoS"):
        return pred_class
    st = attack_srcs_per_target[key]
    if now_s > 0 and (now_s - st.get("win_start", now_s)) > VOLUMETRIC_WINDOW_SECONDS:
        st["srcs"] = set()
        st["win_start"] = now_s
    if is_fwd:
        st["srcs"].add(src_for_set)
    n_unique = len(st["srcs"])
    if pred_class == "DDoS" and n_unique < DDOS_MIN_UNIQUE_SRC:
        return "DoS"     # tek kaynak -> DDoS degil DoS
    if pred_class == "DoS" and n_unique >= DDOS_MIN_UNIQUE_SRC:
        return "DDoS"    # cok kaynak -> DDoS
    return pred_class


# ── (R37/D3 parite) R34 KONSEY-MUAFIYETININ SINIF KAPSAMI ─────────────────
# Kanonik gerekce/olcum: process_pcap_multiclass.py'deki
# COMMITTEE_WAIVER_EXEMPT_CLASSES yorum blogu. Ozet: R34'un konsey guvencesi
# tum siniflara kosulsuz uygulaniyordu ama kanit tabani (138 DoS-Hulk vs 12
# canli mesru baglanti) yalnizca DoS uzerinde olculmustu; ESP32 canli-atesinde
# 1016 gercek FTP-Patator paketi (lstm 0.85-0.944, susp=1, floor 0.95) bu
# yuzden sessizce BENIGN'e dustu. Degisiklik bir GEVSETMEdir -> TP monoton
# artar/sabit kalir, ASLA azalmaz (reddedilen MIN_SUSP=2'nin TERSI yon).
COMMITTEE_WAIVER_EXEMPT_CLASSES = (
    frozenset({"FTP-Patator", "SSH-Patator"})
    if os.environ.get("MC_R37_D3", "1") != "0" else frozenset()
)


def _is_cdn_source(src_dst_universe, src_ip, dst_ip, now_s):
    """(R27 parite) Strateji A, kayan pencere -- bkz. yukarıdaki yorum."""
    st = src_dst_universe[src_ip]
    if now_s > 0 and (now_s - st.get("win_start", now_s)) > VOLUMETRIC_WINDOW_SECONDS:
        st["dsts"] = set()
        st["win_start"] = now_s
    st["dsts"].add(dst_ip)
    return len(st["dsts"]) >= CDN_DST_THRESHOLD


def _second_best_class(proba, benign_idx, classes):
    """(R27/O3 parite -- bkz. process_pcap_multiclass.py Strateji G) argmax
    BENIGN olsa da risk_score esigi gectiginde gosterilecek en olasi IKINCI
    (BENIGN olmayan) sinifi dondurur. Eskiden bu betikte volumetrik dalda
    boyle bir durumda is_alert=True iken etiket celiskili sekilde "Tehdit
    Yok (BENIGN)" kaliyordu -- canli hat bunu zaten duzeltmisti."""
    order = np.argsort(proba)[::-1]
    for idx in order:
        if idx != benign_idx:
            return classes[idx]
    return classes[benign_idx]


def run_offline(rf, lstm, pipeline, classes, n_classes, benign_idx, rows):
    # Önceden tüm LSTM tahminlerini hesapla (tek geçişte)
    lstm_proba_cache = _precompute_lstm(rows, pipeline, lstm, n_classes, benign_idx)

    flow_table = UniversalRealTimeFlowTableV5()
    results    = []

    # ── Strateji izleme yapilari (R27 parite: kayan pencere) ──────────────
    # Strateji A: src_ip -> gördugu benzersiz dst_ip seti (CDN tespiti)
    src_dst_universe: dict = defaultdict(lambda: {"dsts": set(), "win_start": 0.0})
    # Strateji D: (dst_ip, dst_port) -> saldiran benzersiz src_ip seti
    attack_srcs_per_target: dict = defaultdict(lambda: {"srcs": set(), "win_start": 0.0})

    print(f"\n{'='*70}")
    print("Multiclass Offline Test + Strateji katmani (Gecis-2)...")
    print(f"{'='*70}\n")

    for i, packet in enumerate(rows):
        try:
            # (R27/O3 parite -- bkz. process_pcap_multiclass.py) IP katmani
            # olmayan paketleri (ARP/saf L2) dusur -- canli hat R17'den beri
            # bunlari flow-table'a hic sokmuyordu, bu betik sokuyordu.
            if _is_non_ip(packet):
                continue

            packet_no = packet.get("frame.number", i)
            src_ip    = _get_src_ip(packet)
            dst_ip    = _get_dst_ip(packet)
            flags     = packet.get("tcp.flags", "0x0000")

            now_s = _to_float(packet.get("frame.time_epoch"), 0.0)

            # -- Strateji A: src'nin gordugu tum dst'leri kaydet (R27: kayan pencere) --
            is_cdn_src = _is_cdn_source(src_dst_universe, src_ip, dst_ip, now_s)

            if PRINT_EACH:
                print(f"[Paket {packet_no}] {src_ip} -> {dst_ip} | flags={flags}")

            # -- Ozellik cikarimi --
            master_features, flow_key = flow_table.extract_universal_features(packet)
            if master_features is None:
                master_features = {col: 0.0 for col in RAW_COLUMNS_V5}
                master_features["PSH Flag Count"] = 1.0
                master_features["Destination Port"] = 0.0
                flow_key = None

            dst_port   = master_features.get("Destination Port", 0.0)
            rate_value = float(master_features.get(VOLUMETRIC_FEATURE, 0.0))

            # (R37/D2 parite) Strateji-D icin yon-farkindalikli baglam; paket
            # basina BIR kez hesaplanir ve hem disambiguasyon hem RF-override
            # n_srcs sayimi AYNI anahtari kullanir.
            _ddos_key, _ddos_src, _ddos_is_fwd = _ddos_src_context(
                flow_table, flow_key, src_ip, dst_ip, dst_port)

            # -- Dispatcher: volumetrik mi? --
            (is_flood, vol_pkts, vol_syn, vol_ratio, is_scan, distinct_ports,
             is_conn_flood, distinct_conns) = flow_table.update_volumetric(packet)

            # (R33 parite) rate sinyali SABIT-PAYDALI pencereden gelir.
            if R33_RATE_WINDOW:
                rate_win_pkts = (flow_table.update_rate_window(flow_key, now_s)
                                 if flow_key is not None else 0)
                rate_trigger = rate_win_pkts >= RATE_WIN_MIN_PKTS
            else:
                flow_total_pkts = 0
                if flow_key is not None and flow_key in flow_table.flows:
                    _f = flow_table.flows[flow_key]
                    flow_total_pkts = _f.get("fwd_packets", 0) + _f.get("bwd_packets", 0)
                rate_trigger = (rate_value > VOLUMETRIC_THRESHOLD) and (flow_total_pkts >= 2)
            # (R28/O1.3 parite) dorduncu volumetrik tetikleyici.
            is_volumetric = bool(is_flood or is_scan or rate_trigger or is_conn_flood)

            # -- R9: DNS istisna --
            _udp_sport     = _to_int(packet.get("udp.srcport"), 0)
            _udp_dport     = _to_int(packet.get("udp.dstport"), 0)
            _tcp_flags_int = _parse_tcp_flags(packet.get("tcp.flags"))
            is_dns_packet  = (_udp_sport == 53 or _udp_dport == 53) and _tcp_flags_int == 0
            if is_dns_packet and is_volumetric:
                is_volumetric = False

            # -- Ortak: scaled feature vektoru --
            row_df = pd.DataFrame(
                [{col: float(master_features.get(col, 0.0)) for col in RAW_COLUMNS_V5}],
                columns=RAW_COLUMNS_V5,
            )
            scaled_input = _sanitize_matrix(pipeline.transform(row_df))

            counter_score = None
            is_ood, ood_dist, ood_pval = False, None, None  # (R19/R20/S5 parite) yalnizca non-volumetrik dalda degerlendirilir
            is_committee_veto = False  # (R26 Faz 1 parite) yalnizca non-volumetrik dalda degerlendirilir

            if is_volumetric:
                # (R29/O4 parite) VOLUMETRIK dalda OOD SALT-GOZLEM olcumu:
                # mesafe hesaplanir/raporlanir ama is_ood False KALIR
                # (karar deterministik hacimsel imzada). Uretim hattiyla
                # birebir ayni davranis -- bkz. process_pcap_multiclass.py.
                ood_dist = ood_distance(np.asarray(scaled_input)[0])
                ood_pval = ood_pvalue(ood_dist)

                # ══ VOLUMETRİK: MulticlassRF ══ (R27/Y2 parite: ONNX-oncelikli, joblib-yedekli)
                rf_proba = _rf_onnx_proba_or_none(scaled_input[0])
                if rf_proba is None:
                    rf_proba = rf.predict_proba(scaled_input)[0]
                p_benign_rf   = float(rf_proba[benign_idx])
                rf_risk       = 1.0 - p_benign_rf
                rf_class_idx  = int(np.argmax(rf_proba))
                rf_class_name = classes[rf_class_idx]

                # -- Strateji D: DoS/DDoS disambiguasyon (RF icin) --
                rf_class_name = _apply_dos_ddos_disambiguate(
                    rf_class_name, _ddos_key, _ddos_src, _ddos_is_fwd,
                    attack_srcs_per_target, now_s)

                if is_flood:
                    # (R28/O1.1 parite) iki kollu guven.
                    flood_conf = float(max(
                        vol_ratio * min(vol_syn / (SYN_BURST_MIN * 3), 1.0),
                        min(vol_syn / SYN_ABS_FULL, 1.0) if vol_syn >= SYN_ABS_MIN else 0.0,
                    ))
                    risk_score = max(rf_risk, flood_conf)

                    # -- Strateji C: Flood+RF=BENIGN -> kaynak sayisina gore relabel --
                    if rf_class_name == "BENIGN":
                        n_srcs = len(attack_srcs_per_target.get(_ddos_key, {src_ip}))
                        flood_label = "DoS" if n_srcs < DDOS_MIN_UNIQUE_SRC else "DDoS"
                        # (R37/D1 parite) port-farkindalikli ad.
                        detected_attack_name = _flood_override_name(
                            "Hacimsel/SYN-Flood", flood_label, dst_port)
                    else:
                        detected_attack_name = f"Hacimsel/SYN-Flood -> {rf_class_name}"

                elif is_scan:
                    # -- Strateji C: Scan+RF=BENIGN -> alert uretme --
                    # (R28/O1.2 pariteti) KOSULSUZ veto -> distinct_ports-KAPILI.
                    # Gerekce/olcum process_pcap.py SCAN_RF_VETO_MAX_PORTS
                    # yorumunda: benign port cesitliligi onlarca (p75=77),
                    # gercek tarama binlerce (p25=997) porta dokunur. Kosulsuz
                    # hali bu slice'ta 120 GERCEK ters-yon DoS paketini
                    # bastiriyordu (O3.2'nin cift-yonlu GT'siyle gorunur oldu).
                    if rf_class_name == "BENIGN" and distinct_ports < SCAN_RF_VETO_MAX_PORTS:
                        scan_conf = float(min(
                            max(distinct_ports - PORT_SCAN_MIN_PORTS, 0) /
                            (PORT_SCAN_FULL_PORTS - PORT_SCAN_MIN_PORTS), 1.0))
                        # RF BENIGN diyor; sadece scan imzasi var; RF'e guvenerek bastir
                        risk_score = min(rf_risk, ALERT_THRESHOLD - 1e-3)
                        detected_attack_name = "Tehdit Yok (Scan+RF:BENIGN bastiridi)"
                    else:
                        scan_conf = float(min(
                            max(distinct_ports - PORT_SCAN_MIN_PORTS, 0) /
                            (PORT_SCAN_FULL_PORTS - PORT_SCAN_MIN_PORTS), 1.0))
                        risk_score = max(rf_risk, scan_conf)
                        # (R28/O1.2) RF BENIGN diyor AMA port cesitliligi gercek-tarama
                        # bandinda (>=SCAN_RF_VETO_MAX_PORTS) -> imza RF'i ezer. Etiketi
                        # "BENIGN" birakmak celiskili olurdu (alarm var, sinif BENIGN);
                        # hacimsel daldaki RF-override deseniyle ayni sekilde imzanin
                        # kendi sinifi (PortScan) kullanilir.
                        if rf_class_name == "BENIGN":
                            detected_attack_name = "Port-Tarama -> PortScan (RF-override)"
                        else:
                            detected_attack_name = f"Port-Tarama -> {rf_class_name}"
                else:
                    risk_score = rf_risk
                    if rf_class_name == "BENIGN" and risk_score >= ALERT_THRESHOLD:
                        # (R27/O3 parite) Strateji G: argmax BENIGN ama risk_score
                        # (olasilik kutlesi baska siniflara dagilmis) esigi gecti --
                        # eskiden bu durumda etiket celiskili sekilde "Tehdit Yok
                        # (BENIGN)" kalirken is_alert=True oluyordu.
                        _second = _second_best_class(rf_proba, benign_idx, classes)
                        detected_attack_name = f"Hacimsel Anomali -> {_second} (ikincil-sınıf, argmax=BENIGN)"
                    elif rf_class_name == "BENIGN":
                        detected_attack_name = "Tehdit Yok (BENIGN)"
                    else:
                        detected_attack_name = f"Hacimsel Anomali -> {rf_class_name}"

                model_used = "MulticlassRF"

            else:
                # ══ NON-VOLUMETRİK: MulticlassLSTM + Progressive Counter ══
                model_used = "MulticlassLSTM+Counter"

                lstm_proba_raw = lstm_proba_cache.get(i)
                if lstm_proba_raw is None:
                    lstm_proba_raw = np.zeros(n_classes, dtype="float32")
                    lstm_proba_raw[benign_idx] = 1.0

                # (R26 Faz 2 parite) KARAR VEKTORU: soft-vote acikken
                # LSTM+RF+HGB ortalamasi, kapaliyken saf LSTM. Asagidaki TUM
                # kapilar (floor'lar, early_attack, K1, Strateji E/F)
                # degistirilmeden bu vektor uzerinden calisir -- kasitli:
                # "sadece karar vektoru degisti" izole olcumu icin.
                _sv = softvote_proba(lstm_proba_raw, scaled_input[0])
                decision_proba = lstm_proba_raw if _sv is None else _sv
                # (R26 Faz 2 / EARLY-RAW parite) bkz. process_pcap_multiclass.py
                # MC_R26_SOFTVOTE_EARLY_RAW yorum blogu.
                raw_lstm_score = 1.0 - float(lstm_proba_raw[benign_idx])

                p_benign_lstm   = float(decision_proba[benign_idx])
                lstm_class_idx  = int(np.argmax(decision_proba))
                lstm_class_name = classes[lstm_class_idx]
                lstm_score      = 1.0 - p_benign_lstm   # sinif-agnostik skor (degismedi)

                # -- Strateji A: CDN kaynagi + DDoS tahmini -> skoru hafifce bastir --
                # CDN sunucusu (cok farkli dst'ye konusan) DDoS saldirisi yapmaz;
                # skoru %30 indirgemek yetiskinlik kapisi altina dusmesini saglar.
                if lstm_class_name == "DDoS" and is_cdn_src:
                    lstm_score = lstm_score * 0.7

                # -- Strateji D: DoS/DDoS disambiguasyon (LSTM icin) --
                lstm_class_name = _apply_dos_ddos_disambiguate(
                    lstm_class_name, _ddos_key, _ddos_src, _ddos_is_fwd,
                    attack_srcs_per_target, now_s)

                # Progressive counter
                if flow_key is not None and flow_key in flow_table.flows:
                    counter_score, susp_pkts, prog_pkts, susp_ratio = \
                        flow_table.update_progressive(packet, flow_table.flows[flow_key])
                else:
                    counter_score, susp_pkts, prog_pkts, susp_ratio = 0.0, 0, 0, 0.0

                blended    = LSTM_WEIGHT * lstm_score + COUNTER_WEIGHT * counter_score
                risk_score = float(max(lstm_score, counter_score, blended))

                # -- Strateji B (revize): Sinif ozgu olgunluk kapisi esigi --
                # DDoS icin esik 0.98 -> CDN trafiklerini (susp_pkts=0) kapida tutar;
                # gercek DoS-Hulk susp_pkts biriktirir -> early_attack=True -> kapi atlanir.
                effective_lstm_floor = ACTIVE_FLOOR_MAP.get(lstm_class_name, ACTIVE_DEFAULT_FLOOR)

                immature_flow      = prog_pkts < PROG_MIN_PKTS and counter_score < ALERT_THRESHOLD
                _ea_score = (raw_lstm_score if MC_R26_SOFTVOTE_EARLY_RAW else lstm_score)
                _ea_floor = (EARLY_ATTACK_LSTM if MC_R26_SOFTVOTE_EARLY_RAW else ACTIVE_EARLY_ATTACK_LSTM)
                early_attack       = (susp_pkts >= EARLY_ATTACK_MIN_SUSP) and (_ea_score >= _ea_floor)
                # (R34 parite) KONSEY GUVENCESI -- bkz. committee_waives_early_attack()
                # (R37/D3 parite) SINIF KAPSAMI -- bkz. COMMITTEE_WAIVER_EXEMPT_CLASSES.
                if (early_attack
                        and lstm_class_name not in COMMITTEE_WAIVER_EXEMPT_CLASSES
                        and committee_waives_early_attack(scaled_input[0], benign_idx)):
                    early_attack = False
                init_win_fwd       = float(master_features.get("Init_Win_bytes_forward", 0.0))
                # (R13/R17/R18 parite -- bkz. process_pcap.py / process_pcap_multiclass.py)
                # R17 flip'inden sonra "ilk gorulen paket" artefakti pp>1'de de
                # olusabilir (flip edilmis akiste sunucu yonu zaten sayilmistir);
                # esdeger kosul flip edilmis akiste "ilk forward paket"tir.
                _flow_now     = flow_table.flows.get(flow_key) if flow_key is not None else None
                _fwd_pkts_now = _flow_now.get("fwd_packets", 0) if _flow_now is not None else 0
                _first_client_pkt = (prog_pkts <= 1) or (
                    _flow_now is not None and _flow_now.get("flipped") and _fwd_pkts_now <= 1
                )
                client_established = _first_client_pkt and init_win_fwd <= INIT_WIN_CLIENT_MAX
                # (R28/O3.1 parite) R13 (client_established) = UYDURMA vektor
                # -> sert clamp korunur; R7/R8 (kanit zayif ama yok degil) ->
                # olgunlukla olceklenen surekli ceza.
                # (R28/O3.1) surekli olgunluk cezasi OLCULDU ve REDDEDILDI.
                if immature_flow and (not early_attack) and                    (lstm_score < effective_lstm_floor or client_established):
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # -- (R18 parite) YONU DOGRULANMAMIS AKIS KAPISI: flip edilmis
                # akiste forward taraftan HENUZ hic paket gorulmemisse (fwd==0),
                # oznitelik vektorunun forward yarisi UYDURMADIR -> risk esik-alti.
                if _flow_now is not None and _fwd_pkts_now == 0:
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # -- (R23 parite) MULTICAST HEDEF ISTISNASI --
                if _is_multicast_ip(dst_ip):
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # -- (R32/R52 parite) SINIRLI-YAYIN + ALT-AG YAYINI HEDEF ISTISNASI --
                if _is_broadcast_ip(dst_ip, src_ip):
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # -- (R24/K3 parite) PORT-BUCKET DELIL-UYUMU KAPISI -- bkz.
                # process_pcap_multiclass.py yorumu. Sadece SSH-Patator/
                # FTP-Patator (ablasyonda tam collapse gosterdi); WebAttack
                # AYRI mekanizma oldugu icin BILEREK KAPSAM DISI.
                if lstm_class_name == "SSH-Patator" and master_features.get("PortBucket_ssh", 0.0) != 1.0:
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)
                elif lstm_class_name == "FTP-Patator" and master_features.get("PortBucket_ftp", 0.0) != 1.0:
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # -- Strateji E: Dusuk-SYN-sayisi kapisi (DoS/DDoS icin) --
                if lstm_class_name in ("DoS", "DDoS") and not is_flood and vol_syn < 20:
                    if not early_attack:
                        risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # -- Strateji F: WebAttack mutlak skor esigi --
                # Floor sadece immature_flow'a uygulaniyor; olgun CDN akislari bypass ediyor.
                # WebAttack icin olgunluktan bagimsiz mutlak esik: lstm_score < 0.97 ise bastir.
                # Gercek WebAttack saldirilari bu sinif icin yuksek LSTM skoru uretir;
                # CDN overlapi cok daha dusuk skorla giriyor.
                # MC1: "Bot" kapsam disi birakildigi icin bu kapidan cikarildi.
                if lstm_class_name in ("WebAttack",):
                    abs_floor = ACTIVE_FLOOR_MAP.get(lstm_class_name, ACTIVE_DEFAULT_FLOOR)
                    if lstm_score < abs_floor and not early_attack:
                        risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # (R24/K1 parite -- bkz. process_pcap_multiclass.py) ARGMAX-TUTARLILIK
                # KAPISI: argmax=BENIGN iken olasilik kutlesi digerlerine dagilip
                # risk_score esigi gecebiliyordu -- bu betikte onceden bu durum
                # "Zamansal Anomali -> LSTM: BENIGN" gibi celiskili bir etikete
                # yol aciyordu (alert=True ama etiket BENIGN diyor). Modelin kendi
                # argmax kararina ters bir alarm ureteceginden bastirilir.
                if lstm_class_name == "BENIGN":
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # (R26 Faz 1 parite -- bkz. process_pcap_multiclass.py)
                # DAVRANIŞSAL-DAL KONSEY VETOSU: RF VE HGB ikisi de BENIGN'e
                # guvenle katiliyorsa (LSTM'in kararina acikca celisiyorlarsa)
                # bastir. Yalnizca hala esigin ustunde kalan paketlerde cagrilir.
                is_committee_veto = False
                if risk_score >= ALERT_THRESHOLD and committee_dissents(scaled_input[0], benign_idx):
                    is_committee_veto = True
                    risk_score = min(risk_score, ALERT_THRESHOLD - 1e-3)

                # (R19/R20 parite) OOD/kapsam-disi kapisi -- risk_score'a
                # DOKUNULMAZ, sadece raporlanan etiket degisir.
                ood_dist = ood_distance(np.asarray(scaled_input)[0])
                ood_pval = ood_pvalue(ood_dist)
                is_ood = (ood_dist is not None) and (ood_dist > OOD_DISTANCE_THRESHOLD)

                if is_ood:
                    detected_attack_name = "Kapsam Dışı / Bilinmeyen Trafik (OOD)"
                else:
                    detected_attack_name = (
                        f"Zamansal Anomali -> LSTM: {lstm_class_name}"
                        if risk_score >= ALERT_THRESHOLD
                        else "Tehdit Yok (BENIGN)"
                    )

            is_alert = risk_score >= ALERT_THRESHOLD
            if PRINT_EACH:
                tag = "ALERT" if is_alert else "normal"
                print(f"  [{tag}] risk={risk_score:.3f} | {detected_attack_name}")
                print("-" * 60)

            results.append({
                "packet_no":    int(packet_no) if str(packet_no).isdigit() else i,
                "src_ip":       src_ip,
                "dst_ip":       dst_ip,
                "dst_port":     dst_port,
                "is_volumetric": is_volumetric,
                # ── (GATE-SC-0, 2026-08-31) SALT GOZLEM ENSTRUMANTASYONU ──
                # Bu dort alan KARARI ETKILEMEZ; yalnizca sonuc satirina
                # kopyalanir. Amac: davranissal dalda ESIK-ALTI yapisal
                # capraz-akis delilinin kayip saldirilari (FN) ayirt edip
                # etmedigini olcmek. Sayaclar zaten yukarida (satir ~704)
                # her paket icin kosulsuz hesaplaniyor -> ek maliyet SIFIR.
                # Gerekce/kill-kriteri: agent-memory/ids-generalization-
                # strategist/project_structural_corroboration_escalation_
                # 2026_08_31.md. Hacim sayaclari (rate_win_pkts/vol_pkts)
                # ON-KAYITLI OLARAK IZGARA DISI (R30/R32 sensor artefakti).
                "sc_vol_syn":       int(vol_syn),
                "sc_vol_ratio":     round(float(vol_ratio), 4),
                "sc_distinct_ports": int(distinct_ports),
                "sc_distinct_conns": int(distinct_conns),
                "model":        model_used,
                "risk_score":   round(risk_score, 4),
                "counter_score": round(counter_score, 4) if counter_score is not None else None,
                "prediction":   detected_attack_name,
                "is_alert":     is_alert,
                "is_ood":       is_ood,
                "ood_distance": round(ood_dist, 3) if ood_dist is not None else None,
                "ood_pvalue":   round(ood_pval, 6) if ood_pval is not None else None,
                "is_committee_veto": is_committee_veto,
                # (R29/O4) mesafe artik iki dalda da dolu; bu bayrak
                # "karara etki etti mi" ayrimini tasir.
                "ood_gate_applied": (not is_volumetric),
            })

        except Exception as e:
            print(f"[HATA] Paket {i}: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# Ozet istatistikler
# ---------------------------------------------------------------------------
# canli_trafik_kaydi.csv icin bilinen saldiri 5-tuplet
ATTACK_START_PKT = 2648
ATTACK_SRC_IP    = "172.16.0.1"
ATTACK_DST_IP    = "192.168.10.50"
ATTACK_DST_PORT  = 80.0


def print_summary(results, classes):
    df = pd.DataFrame(results)
    n_total  = len(df)
    n_alert  = int(df["is_alert"].sum())
    n_normal = n_total - n_alert

    print(f"\n{'='*70}")
    print("MULTICLASS OFFLINE TEST OZETI (Strateji katmani aktif)")
    print(f"{'='*70}")
    print(f"  Toplam paket     : {n_total:,}")
    print(f"  Alert (>=0.4)    : {n_alert:,}  ({100*n_alert/n_total:.1f}%)")
    print(f"  Normal (<0.4)    : {n_normal:,}  ({100*n_normal/n_total:.1f}%)")
    if "is_ood" in df.columns:
        n_ood = int(df["is_ood"].sum())
        print(f"  OOD/Kapsam disi  : {n_ood:,}  ({100*n_ood/n_total:.1f}%)  [risk_score/is_alert'e dokunulmadi, sadece etiket]")
    if "is_committee_veto" in df.columns:
        n_veto = int(df["is_committee_veto"].sum())
        print(f"  Konsey vetosu (R26 Faz 1): {n_veto:,}  ({100*n_veto/n_total:.1f}%)  [RF+HGB ikisi de BENIGN>=0.95 dedi]")

    # -- Gercek 5-tuple ground-truth (CIFT YONLU) --------------------------
    # (R28/O3.2 -- OLCUM HIJYENI DUZELTMESI) ESKI TANIM TEK YONLUYDU:
    #     src == 172.16.0.1 AND dst == 192.168.10.50 AND dst_port == 80
    # Bir TCP saldirisi cift yonludur; kurbandan saldirgana donen paketler de
    # AYNI saldiri akisinin parcasidir. Tek yonlu tanimda sistemin DOGRU
    # yakaladigi ters-yon saldiri paketleri "benign iken alarm verildi" diye
    # FP sayiliyordu -> FAR yapay olarak sisiyordu. AYNI CSV, AYNI KOSU:
    #     tek yonlu (eski) : TP=2642 FP=319 FN=0   DR=1.0000 FAR=0.0434 MCC=0.9239
    #     cift yonlu (yeni): TP=2951 FP=10  FN=120 DR=0.9609 FAR=0.0014 MCC=0.9695
    # -> FAR 31 KAT sisikmis. Ayrica ikili hattin harness'i
    # (analysis/validate_offline_predictions.py::gt_family, ~satir 94) ZATEN
    # cift yonlu tanim kullaniyordu; iki hattin FAR'lari bu yuzden
    # KIYASLANAMIYORDU. Artik iki harness AYNI GT konvansiyonunu kullanir.
    # NOT: DR'nin 1.0000'dan 0.9609'a dusmesi REGRESYON DEGIL -- eskiden hic
    # sayilmayan 429 ters-yon saldiri paketi artik paydada; bunlarin 120'sini
    # "Scan+RF:BENIGN" kapisi bastiriyor (gercek, olculebilir bir tespit
    # acigi -- eski tanim onu GIZLIYORDU).
    _pair_ok = (
        ((df["src_ip"].astype(str) == ATTACK_SRC_IP) & (df["dst_ip"].astype(str) == ATTACK_DST_IP))
        | ((df["src_ip"].astype(str) == ATTACK_DST_IP) & (df["dst_ip"].astype(str) == ATTACK_SRC_IP))
    )
    is_attack_pkt = (df["packet_no"] >= ATTACK_START_PKT) & _pair_ok & (
        df["dst_port"].astype(float) == ATTACK_DST_PORT
    )
    tp_count = int(df[is_attack_pkt & df["is_alert"]].shape[0])
    fn_count = int(df[is_attack_pkt & ~df["is_alert"]].shape[0])
    fp_count = int(df[~is_attack_pkt & df["is_alert"]].shape[0])
    tn_count = int(df[~is_attack_pkt & ~df["is_alert"]].shape[0])
    n_real_attack = tp_count + fn_count
    n_real_benign = fp_count + tn_count

    far  = fp_count / n_real_benign if n_real_benign else 0
    dr   = tp_count / n_real_attack if n_real_attack else 0
    prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else 0
    mcc_num = tp_count * tn_count - fp_count * fn_count
    mcc_den = ((tp_count + fp_count) * (tp_count + fn_count) *
               (tn_count + fp_count) * (tn_count + fn_count)) ** 0.5
    mcc = mcc_num / mcc_den if mcc_den else 0.0

    print(f"\n  --- Ground-Truth (5-tuple CIFT YONLU: {ATTACK_SRC_IP}<->{ATTACK_DST_IP}:{int(ATTACK_DST_PORT)}, pkt>={ATTACK_START_PKT}) ---")
    print(f"  Saldiri paketleri: {n_real_attack:,} | TP={tp_count} | FN={fn_count} | DR={dr:.4f}")
    print(f"  Benign paketleri : {n_real_benign:,} | FP={fp_count} | TN={tn_count} | FAR={far:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  MCC              : {mcc:.4f}")

    print("\n  Dispatcher dagilimi:")
    for model, cnt in df["model"].value_counts().items():
        print(f"    {model}: {cnt:,}  ({100*cnt/n_total:.1f}%)")

    print("\n  Tespit edilen saldiri kategorileri (alert paketler):")
    alert_df = df[df["is_alert"]]
    if len(alert_df):
        for pred, cnt in alert_df["prediction"].value_counts().items():
            print(f"    {pred}: {cnt:,}  ({100*cnt/len(alert_df):.1f}%)")
    else:
        print("    Hic alert uretilmedi.")

    print("\n  Risk skoru dagilimlari:")
    print(f"    Ortalama : {df['risk_score'].mean():.4f}")
    print(f"    Medyan   : {df['risk_score'].median():.4f}")
    print(f"    Max      : {df['risk_score'].max():.4f}")

    print("\n  En cok alert ureten akislar (src->dst:port):")
    alert_df2 = df[df["is_alert"]].copy()
    alert_df2["pair"] = (alert_df2["src_ip"].astype(str) + " -> " +
                         alert_df2["dst_ip"].astype(str) + ":" +
                         alert_df2["dst_port"].astype(str))
    for pair, cnt in alert_df2["pair"].value_counts().head(10).items():
        print(f"    {pair}: {cnt:,}")

    return df


N_FEAT = len(RAW_COLUMNS_V5)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rf, lstm, pipeline, classes, n_classes, benign_idx = load_multiclass_models()
    rows    = load_packets(CSV_PATH, max_packets=MAX_PACKETS)
    results = run_offline(rf, lstm, pipeline, classes, n_classes, benign_idx, rows)
    df_out  = print_summary(results, classes)

    out_csv = os.path.join(THIS_DIR, "data", "offline_test_multiclass_results.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[+] Sonuclar kaydedildi: {out_csv}")
