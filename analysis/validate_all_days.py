# -*- coding: utf-8 -*-
"""
validate_all_days.py
=====================
offline_test.ipynb / pcap_validation_pilot_v2.py dispatcher'ini (MainRF_v2 +
MainLSTM_v2, SYN-burst -> RF / aksi -> LSTM+Progressive Counter,
update_progressive YENI tasarim) CICIDS-2017'nin TUM gunlerindeki (Tuesday,
Thursday, Friday + Wednesday zaten ayri betiklerle test edildi) saldiri
turleri uzerinde calistirir.

Iki soruyu yanitlar:
  (a) ANOMALI: paket-bazli DR / FAR / precision / MCC (binary attack/benign)
  (b) ILGILI SALDIRI: dispatcher'in uretttigi 'detected_attack_name' (4
      kategori: Tehdit Yok / Hacimsel Esik Ihlali (SYN-Flood Imzasi) /
      Coklu-Port Tarama Imzasi / Hacimsel Anomali (RF Siniflandirici) /
      Davranissal/Sirali Anomali (LSTM Tespiti)) GT saldiri turuyle
      (FTP-Patator, SSH-Patator, WebAttack,
      Infiltration, Bot, PortScan, DDoS) ne kadar ortusuyor
      (dispatch_vs_gt cross-tab).

GROUND TRUTH (TrafficLabelling/*.pcap_ISCX.csv'den, 5-tuple):
  Tuesday   FTP-Patator  172.16.0.1 -> 192.168.10.50:21
  Tuesday   SSH-Patator  172.16.0.1 -> 192.168.10.50:22
  Thursday  WebAttack    172.16.0.1 -> 192.168.10.50:80 (BruteForce/XSS/SQLi)
  Thursday  Infiltration 192.168.10.8 <-> 205.174.165.73:444
  Friday    Bot          205.174.165.73 <-> 192.168.10.x (herhangi port)
  Friday    PortScan     172.16.0.1 <-> 192.168.10.50 (herhangi port)
  Friday    DDoS         172.16.0.1 -> 192.168.10.50:80

BATCH SECIMI: sparse tshark taramasi (_scan_tuesday.py/_scan_thursday.py/
_scan_friday.py) ile saldiri trafiginin yogun oldugu batch'ler bulundu.

Cikti:
    validate_all_days_results.csv    (paket-bazli)
    validate_all_days_summary.json   (genel + gun/saldiri-turu bazli metrikler)
"""

import os
import json
import math
import glob
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

import pcap_validation_pilot as v1
import pcap_validation_pilot_v2 as v2mod

PROJECT_DIR = v1.PROJECT_DIR

# MODEL_VERSION=v3/v4 -> MainRF_vN/MainLSTM_vN (coklu-gun egitim), aksi halde v2.
PCAP_ROOT = os.environ.get("CICIDS_PCAP_ROOT", os.path.join(PROJECT_DIR, "data", "batch_pcaps"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v2")
_suffix = "" if MODEL_VERSION == "v2" else f"_{MODEL_VERSION}"
OUT_CSV = os.path.join(PROJECT_DIR, "results", f"validate_all_days_results{_suffix}.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "results", f"validate_all_days_summary{_suffix}.json")

# GROUPS_FILTER="Thursday/Infiltration,Friday/Bot" -> sadece bu grup(lar)i calistirir.
GROUPS_FILTER = os.environ.get("GROUPS_FILTER", "").strip()

# Gelistirme ortamina ozel yollar: NiFi'nin tshark ile parcaladigi ham
# Tuesday/Thursday/Friday .pcap batch'lerinin bulundugu dizinler (ham
# CICIDS-2017 capture'lari boyut nedeniyle teslim paketine dahil edilmemistir).
BATCH_DIRS = {
    "Tuesday": (os.path.join(PCAP_ROOT, "Tuesday-WorkingHours"), "Tuesday-WorkingHours"),
    "Thursday": (os.path.join(PCAP_ROOT, "Thursday-WorkingHours"), "Thursday-WorkingHours"),
    "Friday": (os.path.join(PCAP_ROOT, "Friday_WorkingHours"), "Friday-WorkingHours"),
    # (R26 Faz 2 / Adim 1 -- 2026-08-14) Wednesday EKLENDI. Dizin adindaki
    # kucuk 'w' (workingHours) KASITLI: diskteki gercek ad bu.
    "Wednesday": (os.path.join(PCAP_ROOT, "Wednesday-workingHours"), "Wednesday-workingHours"),
}


def find_batch_day(day, idx):
    pcap_dir, prefix = BATCH_DIRS[day]
    pat = os.path.join(pcap_dir, f"{prefix}_{idx}_*.pcap")
    matches = glob.glob(pat)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# GROUND-TRUTH FONKSIYONLARI (5-tuple, gt -> (is_attack, family_label))
# ---------------------------------------------------------------------------
def make_pair_port_gt(ip_a, ip_b, ports, family):
    """ports: set[int] icin pair+port eslesmesi; None -> sadece pair (herhangi port)."""
    ports_set = set(ports) if ports else None

    def gt(packet):
        s, d = packet.get("ip.src", ""), packet.get("ip.dst", "")
        sp = v1._to_int(packet.get("tcp.srcport"), 0)
        dp = v1._to_int(packet.get("tcp.dstport"), 0)
        pair = {s, d}
        if pair == {ip_a, ip_b}:
            if ports_set is None or dp in ports_set or sp in ports_set:
                return 1, family
        return 0, "BENIGN"

    return gt


def gt_bot(packet):
    """Friday Bot: 205.174.165.73 <-> 192.168.10.x (herhangi port)."""
    s, d = packet.get("ip.src", ""), packet.get("ip.dst", "")
    C2 = "205.174.165.73"
    if C2 == s and d.startswith("192.168.10."):
        return 1, "Bot"
    if C2 == d and s.startswith("192.168.10."):
        return 1, "Bot"
    return 0, "BENIGN"


def gt_clean(packet):
    return 0, "BENIGN"


# ---------------------------------------------------------------------------
# GUN / GRUP / BATCH TANIMLARI
# ---------------------------------------------------------------------------
GROUPS = [
    # (day, family, [batch_idx...], gt_func)
    ("Tuesday", "FTP-Patator", ["00570", "00630", "00650", "00660"],
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {21}, "FTP-Patator")),
    ("Tuesday", "SSH-Patator", ["00930", "00950", "00990", "01020"],
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {22}, "SSH-Patator")),
    ("Tuesday", "Clean", ["00030", "00800", "01130"], gt_clean),

    ("Thursday", "WebAttack", ["00300", "00350", "00360", "00370"],
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {80}, "WebAttack")),
    ("Thursday", "Infiltration", ["00690", "00700"],
        make_pair_port_gt("192.168.10.8", "205.174.165.73", {444}, "Infiltration")),
    ("Thursday", "Clean", ["00050", "00150", "00920"], gt_clean),

    ("Friday", "Bot", ["00460", "00470"], gt_bot),
    ("Friday", "PortScan", ["00700", "00720", "00750"],
        make_pair_port_gt("172.16.0.1", "192.168.10.50", None, "PortScan")),
    ("Friday", "DDoS", ["00830", "00870", "00920"],
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {80}, "DDoS")),
    ("Friday", "Clean", ["00050", "00200", "00300"], gt_clean),
]

# ── (R26 Faz 2 / Adim 1 -- 2026-08-14) WEDNESDAY/DoS KAPSAM ACIGI ──────────
# BULGU: bu korpusta Wednesday HIC YOKTU -> capraz-gun dogrulamada DoS SINIFI
# ICIN TEK BIR HELD-OUT PAKET BILE OLCULMUYORDU. Multiclass hattin tum DoS
# kaniti tek bir 10k-paketlik canli yakalamaya (`data/canli_trafik_kaydi.csv`,
# DoS-Hulk) dayaniyordu. R26 Faz 2 soft-vote A/B'sinde OLCULEN TEK regresyon
# tam da o dosyadaydi -> tek-korpus artefakti mi, gercek mi, ayirt edilemiyordu.
# Bu blok o boslugu kapatir.
#
# BATCH SECIMI (sizinti hijyeni -- egitim setiyle KESISIM YOK, dogrulandi):
#   * `build_streaming_dataset.TRAIN_BATCHES` (28 batch) = EGITIM. Asagidaki
#     hicbir indeks bu listede DEGIL.
#   * `build_streaming_dataset.VALIDATION_BATCHES` (12 batch) = projenin
#     ONCEDEN ILAN ETTIGI Wednesday held-out kumesi -> hacimsel DoS ve Clean
#     batch'leri BURADAN secildi (00700/00850/01010, 00100/00300/01100).
#   * Yavas-DoS (Slowloris/Slowhttptest) penceresi VALIDATION_BATCHES'te YOK;
#     bu yuzden 00610/00630/00650 ne egitimde ne dogrulamada olan, hic
#     dokunulmamis batch'lerden alindi.
#
# AILE ATAMASI OLCUMLE DOGRULANDI (tshark, 2026-08-14) -- yayinlanmis CICIDS
# saat cizelgesine DEGIL, trafigin KENDI seklinE gore (saat dilimi belirsizligi
# nedeniyle):
#   00610  17.7s  564 pkt/s  saldiri=  661  3.2 pkt/baglanti  -> yavas-DoS
#   00630  10.6s  943 pkt/s  saldiri=  332  2.7 pkt/baglanti  -> yavas-DoS
#   00650  46.3s  216 pkt/s  saldiri=  274  6.2 pkt/baglanti  -> yavas-DoS
#   00700   0.8s 13077 pkt/s saldiri= 9995 12.4 pkt/baglanti  -> Hulk
#   00850   0.7s 14384 pkt/s saldiri= 9943 10.1 pkt/baglanti  -> Hulk
#   01010  22.0s  455 pkt/s  saldiri= 6085  8.5 pkt/baglanti  -> GoldenEye
# Iki rejim AYRI GRUPLARA konuldu: tek "DoS" grubunda ortalanirlarsa yavas-DoS
# (birkac yuz paket) hacimsel DoS'un (~10k paket) yaninda gorunmez olurdu --
# oysa R26 Faz 2'nin bedeli tam olarak OLGUNLASMAMIS/acilis fazi paketlerinde.
#
# GT konvansiyonu diger gunlerle AYNI: pair {172.16.0.1, 192.168.10.50} +
# port 80, cift yonlu (Friday/DDoS ile birebir ayni cagri).
# NOT: Wednesday'in Heartbleed'i (192.168.10.51:444) bu pair/port'a GIRMEZ ->
# GT'ye dahil DEGIL. Bu KASITLI: MC1 kapsaminda Heartbleed sinifi yok
# (Thursday/Infiltration'in kapsam-disi birakilmasiyla ayni gerekce).
WEDNESDAY_GROUPS = [
    ("Wednesday", "DoS-Slow", ["00610", "00630", "00650"],
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {80}, "DoS")),
    ("Wednesday", "DoS-Volumetric", ["00700", "00850", "01010"],
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {80}, "DoS")),
    ("Wednesday", "Clean", ["00100", "00300", "01100"], gt_clean),
]

# VARSAYILAN KAPALI (`INCLUDE_WEDNESDAY=1` ile acilir). GEREKCE: bu betigin ve
# `validate_all_days_multiclass.py`'nin commit'li referans rakamlari
# (results/*_summary.json) mevcut 31 batch uzerinden uretildi; gruplari
# kosulsuz eklemek o rakamlari sessizce gecersiz kilardi ve
# [[r34-early-attack-committee-guard]]'in "hangi olcume ait?" acik maddesini
# tekrarlardi. Kalici hale getirme KULLANICI KARARI.
INCLUDE_WEDNESDAY = os.environ.get("INCLUDE_WEDNESDAY", "0") == "1"
if INCLUDE_WEDNESDAY:
    GROUPS = GROUPS + WEDNESDAY_GROUPS
    print(f"[*] INCLUDE_WEDNESDAY=1 -> Wednesday/DoS gruplari eklendi "
          f"({len(WEDNESDAY_GROUPS)} grup, {sum(len(g[2]) for g in WEDNESDAY_GROUPS)} batch)")


# ---------------------------------------------------------------------------
# DISPATCHER (offline_test.ipynb cell 12 / 736c175f bire bir) - genisletilmis:
# 'detected_attack_name' / 'is_flood' / 'rf_label' de donduruyor.
# process_pcap.py / pcap_validation_pilot.py::dispatch_packet ile R9/R10/R12/R13
# paritesi (2026-06-15 audit, madde 3).
# ---------------------------------------------------------------------------
def dispatch_packet_ext(packet, flow_table, cfg):
    master_features, flow_key = flow_table.extract_universal_features(packet)
    if master_features is None:
        master_features = {col: 0.0 for col in cfg["RAW_COLUMNS"]}
        master_features["PSH Flag Count"] = 1.0
        flow_key = None

    rate_value = float(master_features.get(cfg["VOLUMETRIC_FEATURE"], 0.0))
    (is_flood, vol_pkts, vol_syn, vol_ratio, is_scan, distinct_ports,
     is_conn_flood, distinct_conns) = flow_table.update_volumetric(packet)

    # (R10) tek-paketlik-akis rate yapayligi: rate sinyali sadece akis >=2
    # pakete ulastiginda gecerli (is_flood/is_scan etkilenmez).
    # (R33 parite) rate sinyali artik SABIT-PAYDALI kayan pencereden gelir --
    # gerekce/kalibrasyon pipeline/process_pcap.py R33 yorumunda; toggle
    # PILOT_R33=0 ile R33-oncesi davranis (pcap_validation_pilot.py ile ayni).
    if v1.R33_RATE_WINDOW:
        _now_s = v1._to_float(packet.get("frame.time_epoch"), 0.0)
        rate_win_pkts = (flow_table.update_rate_window(flow_key, _now_s)
                         if flow_key is not None else 0)
        rate_trigger = rate_win_pkts >= v1.RATE_WIN_MIN_PKTS
    else:
        flow_total_pkts = 0
        if flow_key is not None:
            _f = flow_table.flows.get(flow_key)
            if _f is not None:
                flow_total_pkts = _f.get("fwd_packets", 0) + _f.get("bwd_packets", 0)
        rate_trigger = (rate_value > cfg["VOLUMETRIC_THRESHOLD"]) and (flow_total_pkts >= 2)
    # (R28/O1.3) dorduncu volumetrik tetikleyici -- pcap_validation_pilot.py parite.
    is_volumetric = bool(is_flood or is_scan or rate_trigger or is_conn_flood)

    # (R9) DNS/UDP-53 istisnasi: gecmiste is_scan=True olmus (src,dst)
    # ikilisinin sonraki DNS yanit paketleri RF dalina kacmasin.
    _udp_sport = v1._to_int(packet.get("udp.srcport"), 0)
    _udp_dport = v1._to_int(packet.get("udp.dstport"), 0)
    _tcp_flags_int = v1._parse_tcp_flags(packet.get("tcp.flags"))
    is_dns_packet = (_udp_sport == 53 or _udp_dport == 53) and _tcp_flags_int == 0
    if is_dns_packet and is_volumetric:
        is_volumetric = False

    rf_label = None
    counter_score = None

    if is_volumetric:
        row_df = pd.DataFrame(
            [{col: float(master_features.get(col, 0.0)) for col in cfg["RAW_COLUMNS"]}],
            columns=cfg["RAW_COLUMNS"],
        )
        scaled_input = cfg["PIPELINE"].transform(row_df)
        # ONNX RF hizlandirma (Strateji-1): cfg["rf_onnx"] varsa (v5) ONNX,
        # yoksa joblib RF'ye duser -- cikti birebir esdeger.
        rf_proba = v1.rf_predict_proba(cfg, scaled_input)[0]
        p_benign_rf = float(rf_proba[cfg["RF_BENIGN_POS"]])
        rf_risk = float(1.0 - p_benign_rf)
        rf_idx = int(np.argmax(rf_proba))
        rf_label = cfg["RF_CLASSES"][rf_idx]

        # (R28/O1.1-O1.3) pcap_validation_pilot.py::dispatch_packet ile parite.
        if is_flood:
            flood_conf = float(vol_ratio * min(vol_syn / (v1.SYN_BURST_MIN * 3), 1.0))
            if v1.R28_SYN_ABS and vol_syn >= v1.SYN_ABS_MIN:
                flood_conf = max(flood_conf, float(min(vol_syn / v1.SYN_ABS_FULL, 1.0)))
            risk_score = max(rf_risk, flood_conf)
            detected_attack_name = "Hacimsel Esik Ihlali (SYN-Flood Imzasi)"
        elif is_conn_flood:
            # (R28/O1.3) kapi-tutarli rampa -- bkz. pcap_validation_pilot.py.
            conn_conf = float(min(distinct_conns / v1.CONN_FLOOD_FULL, 1.0))
            risk_score = max(rf_risk, conn_conf)
            detected_attack_name = "Baglanti Seli Imzasi (Hedef-Toplulastirmali)"
        elif is_scan:
            scan_conf = float(min(
                max(distinct_ports - v1.PORT_SCAN_MIN_PORTS, 0) /
                (v1.PORT_SCAN_FULL_PORTS - v1.PORT_SCAN_MIN_PORTS), 1.0))
            # (R28/O1.2) RF-veto kapisi -- gerekce process_pcap.py'de.
            if v1.R28_SCAN_RF_VETO and rf_label == cfg["RF_CLASSES"][cfg["RF_BENIGN_POS"]] \
               and distinct_ports < v1.SCAN_RF_VETO_MAX_PORTS:
                scan_conf = 0.0
            risk_score = max(rf_risk, scan_conf)
            detected_attack_name = "Coklu-Port Tarama Imzasi"
        else:
            risk_score = rf_risk
            detected_attack_name = (
                "Tehdit Yok (BENIGN)" if rf_label == 0
                else "Hacimsel Anomali (RF Siniflandirici)"
            )
        model = "RF"
    else:
        row_df = pd.DataFrame(
            [{col: float(master_features.get(col, 0.0)) for col in cfg["RAW_COLUMNS"]}],
            columns=cfg["RAW_COLUMNS"],
        )
        scaled_input = cfg["PIPELINE"].transform(row_df)
        n_feat_expected = cfg["n_feat_lstm"]
        curr_feat = scaled_input.shape[1]
        if curr_feat < n_feat_expected:
            pad = np.zeros((scaled_input.shape[0], n_feat_expected - curr_feat))
            lstm_in_data = np.hstack([scaled_input, pad])
        else:
            lstm_in_data = scaled_input[:, :n_feat_expected]
        lstm_input = lstm_in_data.reshape(1, 1, n_feat_expected).astype("float32")
        raw_pred = np.asarray(cfg["lstm_fast"](lstm_input))[0]
        if len(raw_pred) == 1:
            p_attack = float(raw_pred[0])
            proba = np.array([1.0 - p_attack, p_attack])
        else:
            proba = raw_pred
        lstm_score = float(1.0 - proba[cfg["BENIGN_INDEX"]])

        if flow_key is not None and flow_key in flow_table.flows:
            counter_score, susp_count, prog_pkts, _ = flow_table.update_progressive(packet, flow_table.flows[flow_key])
        else:
            counter_score, susp_count, prog_pkts = 0.0, 0, 0
        blended = v1.LSTM_WEIGHT * lstm_score + v1.COUNTER_WEIGHT * counter_score
        risk_score = float(max(lstm_score, counter_score, blended))

        # (R12/R13) akis-olgunlugu kapisi: immature akiste (prog_pkts<PROG_MIN_PKTS)
        # erken-saldiri (R12: susp_count>=1 & lstm>=0.85) veya kurulu-istemci
        # (R13: prog_pkts<=1 & init_win<=512) istisnasi yoksa risk esik-altina sabitlenir.
        immature_flow = prog_pkts < v1.PROG_MIN_PKTS and counter_score < v1.DISPATCH_ALERT_THRESHOLD
        early_attack = (susp_count >= v1.EARLY_ATTACK_MIN_SUSP) and (lstm_score >= v1.EARLY_ATTACK_LSTM)
        # (R34 parite) konsey guvencesi -- bkz. v1.committee_waives_early_attack()
        if early_attack and v1.committee_waives_early_attack(cfg, scaled_input):
            early_attack = False
        init_win_fwd = float(master_features.get("Init_Win_bytes_forward", 0.0))
        client_established = prog_pkts <= 1 and init_win_fwd <= v1.INIT_WIN_CLIENT_MAX
        # (R28/O3.1) kapi-oncesi ham skor: esik/us kalibrasyonu TEK kosudan
        # offline supurulebilsin diye CSV'ye yazilir.
        precap_risk = risk_score
        # (R28) hangi olgunluk-kapisi kolunun calistigi: us (exponent)
        # kalibrasyonu TEK kosudan offline supurulebilsin diye kaydedilir.
        gate_applied = ""
        # (R28/O3.1) surekli olgunluk cezasi OLCULDU ve REDDEDILDI (bootstrap
        # %95 GA sifiri iceriyor) -- gerekce process_pcap.py'deki yorum blogunda.
        # `gate_applied`/`precap_risk`/`prog_pkts` kolonlari KALDI: gelecekteki
        # kapi kalibrasyonlari tek kosudan offline supurulebilsin diye.
        if immature_flow and (not early_attack) and            (lstm_score < v1.LSTM_ALERT_FLOOR or client_established):
            risk_score = min(risk_score, v1.DISPATCH_ALERT_THRESHOLD - 1e-3)
            gate_applied = "hard_clamp"

        # (R28/O2.1) davranissal-dal konsey vetosu -- pcap_validation_pilot.py
        # ve process_pcap.py ile parite.
        if risk_score >= v1.DISPATCH_ALERT_THRESHOLD and v1.committee_dissents(cfg, scaled_input):
            risk_score = min(risk_score, v1.DISPATCH_ALERT_THRESHOLD - 1e-3)

        detected_attack_name = (
            "Davranissal/Sirali Anomali (LSTM Tespiti)" if risk_score >= v1.ALERT_THRESHOLD
            else "Tehdit Yok (BENIGN)"
        )
        model = "LSTM+Counter"

    return {
        # (R28) olcum/kalibrasyon alanlari -- karar mantigini ETKILEMEZ
        "precap_risk": round(float(precap_risk), 6) if not is_volumetric else None,
        "gate_applied": gate_applied if not is_volumetric else "",
        "prog_pkts": prog_pkts if not is_volumetric else None,
        "lstm_score": round(float(lstm_score), 6) if not is_volumetric else None,
        "counter_score": round(float(counter_score), 6) if counter_score is not None else None,
        "src_ip": packet.get("ip.src", "0.0.0.0"),
        "dst_ip": packet.get("ip.dst", "0.0.0.0"),
        "dst_port": master_features.get("Destination Port"),
        "is_volumetric": is_volumetric,
        "is_flood": is_flood,
        "is_scan": is_scan,
        # (R28/O1.3) yeni sinyal, olcum icin CSV'ye de yaziliyor
        "is_conn_flood": is_conn_flood,
        "distinct_conns": distinct_conns,
        "vol_pkts": vol_pkts,
        "vol_syn": vol_syn,
        "vol_ratio": round(vol_ratio, 4),
        "distinct_ports": distinct_ports,
        "rf_label": rf_label,
        "model": model,
        "risk_score": risk_score,
        "detected_attack_name": detected_attack_name,
    }


# ---------------------------------------------------------------------------
# METRIK YARDIMCILARI (validate_slow_attacks.py ile ayni)
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def metrics_from_conf(tp, fp, tn, fn):
    dr = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return {
        "DR_recall_attack": round(dr, 4), "FAR": round(far, 4),
        "precision_attack": round(prec, 4), "F1_attack": round(f1, 4),
        "MCC": round(mcc, 4), "accuracy": round(acc, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def tp_fp_fn_tn_label(gt, alert):
    if gt == 1 and alert == 1:
        return "TP"
    if gt == 0 and alert == 1:
        return "FP"
    if gt == 1 and alert == 0:
        return "FN"
    return "TN"


# ---------------------------------------------------------------------------
# ANA
# ---------------------------------------------------------------------------
def main():
    flow_table_cls = v1.UniversalRealTimeFlowTable
    if MODEL_VERSION == "v5":
        import pcap_validation_pilot_v5 as v5mod
        cfg = v5mod.load_models_v5()
        flow_table_cls = v5mod.UniversalRealTimeFlowTableV5
    elif MODEL_VERSION == "v4":
        import pcap_validation_pilot_v4 as v4mod
        cfg = v4mod.load_models_v4()
    elif MODEL_VERSION == "v3":
        import pcap_validation_pilot_v3 as v3mod
        cfg = v3mod.load_models_v3()
    else:
        cfg = v2mod.load_models_v2()

    groups = GROUPS
    if GROUPS_FILTER:
        wanted = set(GROUPS_FILTER.split(","))
        groups = [g for g in GROUPS if f"{g[0]}/{g[1]}" in wanted]
        print(f"[*] GROUPS_FILTER aktif: {sorted(wanted)} -> {len(groups)} grup calisacak")

    all_rows = []
    batch_summ = []

    for day, family, batches, gt_func in groups:
        for idx in batches:
            pcap = find_batch_day(day, idx)
            if not pcap:
                print(f"[!] {day} batch {idx} bulunamadi, atlaniyor.")
                continue
            print(f"\n[{day} | grup={family} | batch {idx}] {os.path.basename(pcap)}", flush=True)
            packets = v1.read_pcap(pcap)
            print(f"    {len(packets)} paket okundu, dispatcher calisiyor...", flush=True)

            flow_table = flow_table_cls()
            y_true, y_pred, risks = [], [], []
            n_rf = n_lstm = 0

            for pkt in packets:
                gt, fam = gt_func(pkt)
                res = dispatch_packet_ext(pkt, flow_table, cfg)
                alert = 1 if res["risk_score"] >= v1.ALERT_THRESHOLD else 0
                y_true.append(gt)
                y_pred.append(alert)
                risks.append(res["risk_score"])
                if res["model"] == "RF":
                    n_rf += 1
                else:
                    n_lstm += 1
                all_rows.append({
                    "day": day, "group": family, "batch": idx,
                    "src_ip": res["src_ip"], "dst_ip": res["dst_ip"],
                    "dst_port": res["dst_port"], "model": res["model"],
                    "is_volumetric": res["is_volumetric"], "is_flood": res["is_flood"],
                    "is_scan": res["is_scan"], "vol_pkts": res["vol_pkts"],
                    "vol_syn": res["vol_syn"], "vol_ratio": res["vol_ratio"],
                    "distinct_ports": res["distinct_ports"],
                    # (R28) yeni sinyal + kalibrasyon alanlari
                    "is_conn_flood": res.get("is_conn_flood"),
                    "distinct_conns": res.get("distinct_conns"),
                    "precap_risk": res.get("precap_risk"),
                    "gate_applied": res.get("gate_applied"),
                    "prog_pkts": res.get("prog_pkts"),
                    "lstm_score": res.get("lstm_score"),
                    "counter_score": res.get("counter_score"),
                    "rf_label": res["rf_label"],
                    "risk_score": round(res["risk_score"], 4),
                    "detected_attack_name": res["detected_attack_name"],
                    "gt_attack": gt, "gt_family": fam,
                    "alert": alert, "outcome": tp_fp_fn_tn_label(gt, alert),
                })

            tp, fp, tn, fn = confusion(y_true, y_pred)
            risks_np = np.array(risks) if risks else np.array([0.0])
            bs = {
                "day": day, "group": family, "batch": idx, "pcap": os.path.basename(pcap),
                "total_pkts": len(packets),
                "gt_attack_pkts": sum(y_true), "gt_benign_pkts": len(packets) - sum(y_true),
                "alerts": sum(y_pred),
                "mean_risk": round(float(risks_np.mean()), 4),
                "max_risk": round(float(risks_np.max()), 4),
                "n_RF": n_rf, "n_LSTM": n_lstm,
            }
            bs.update(metrics_from_conf(tp, fp, tn, fn))
            batch_summ.append(bs)
            print(f"    pkts={len(packets)} gt_attack={sum(y_true)} alerts={sum(y_pred)} "
                  f"RF={n_rf} LSTM={n_lstm} | DR={bs['DR_recall_attack']} FAR={bs['FAR']} "
                  f"MCC={bs['MCC']}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    summary = {"models": MODEL_VERSION, "alert_threshold": v1.ALERT_THRESHOLD, "batches": batch_summ}

    # gun + saldiri-turu bazli ozet + dispatch_vs_gt cross-tab
    per_group = {}
    for day, family, _, _ in GROUPS:
        key = f"{day}/{family}"
        if key in per_group:
            continue
        sub = df[(df["day"] == day) & (df["group"] == family)]
        if sub.empty:
            continue
        tp, fp, tn, fn = confusion(sub["gt_attack"], sub["alert"])
        m = metrics_from_conf(tp, fp, tn, fn)
        m["dispatch_vs_gt"] = (
            sub.groupby(["gt_family", "detected_attack_name"]).size()
            .unstack(fill_value=0).to_dict(orient="index")
        )
        per_group[key] = m
    summary["per_group"] = per_group

    # genel (sadece saldiri gruplari, Clean haric)
    attack_df = df[df["group"] != "Clean"]
    tp, fp, tn, fn = confusion(attack_df["gt_attack"], attack_df["alert"])
    summary["overall_attack_groups"] = metrics_from_conf(tp, fp, tn, fn)

    # genel (TUM paketler, Clean dahil)
    tp, fp, tn, fn = confusion(df["gt_attack"], df["alert"])
    summary["overall_all"] = metrics_from_conf(tp, fp, tn, fn)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("GUN / GRUP BAZLI SONUC:")
    for key, m in summary["per_group"].items():
        print(f"  {key:24s}: DR={m['DR_recall_attack']} FAR={m['FAR']} "
              f"precision={m['precision_attack']} MCC={m['MCC']} "
              f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")
    print("\nGENEL (saldiri gruplari):", summary["overall_attack_groups"])
    print("GENEL (tum, Clean dahil):", summary["overall_all"])
    print(f"\n[+] Paket-bazli sonuc : {OUT_CSV}")
    print(f"[+] Ozet (JSON)       : {OUT_JSON}")


if __name__ == "__main__":
    main()
