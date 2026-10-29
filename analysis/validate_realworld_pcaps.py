# -*- coding: utf-8 -*-
"""
validate_realworld_pcaps.py
============================
Bu projenin **GERCEK-DUNYA (CICIDS-DISI) HELD-OUT** degerlendirme harness'i.

NEDEN VAR
---------
`validate_all_days_multiclass.py` modeli CICIDS-2017'nin KENDI batch'lerinde
olcer. Ama bu depodaki her genelleme bulgusu (R35 tek-oturum ezberi, R38
faz-secim yanliligi, R39 manifold probu) ayni seyi soyluyor: **CICIDS-ici
held-out, CICIDS-disi gercek trafik hakkinda cok az sey soyluyor.**

R38/O1 (`r38_xss` retrain adayi) tam olarak bu yuzden GERCEK kanitla
degerlendirilemedi: ESP32 saldirisinin ham verisi halka tamponda donup
gitmisti, aday model yeniden skorlanamadi. Bu betik o acigi kalici olarak
kapatir: **isimlendirilmis, versiyonlanmis, gercek-dunya yakalamalarindan
olusan bir kayit defteri (registry)** tutar ve HERHANGI bir model adayini
(gelecekteki O-C kampanyasiyla egitilmis olan dahil) bu sete karsi
otomatik olarak puanlar.

    +---------------------------------------------------------------+
    |  BU SETLER **YALNIZCA TEST**TIR. ASLA EGITIME KATILMAZ.        |
    +---------------------------------------------------------------+

Gerekce (R39 olcumu, `.claude/agent-memory/ids-generalization-strategist/
project_esp32_real_capture_manifold_probe.md`):
  1. **Port kovasi yanlis.** ESP32 hedefleri :3000 ve :2121 -> ikisi de
     `PortBucket_ephemeral`. Bu bir MODEL GIRDISI. Egitime katmak
     "ephemeral + kisa akis = saldiri" kisayolunu ogretir ve gercek
     :80/:443 web saldirisina TRANSFER ETMEZ (R24 deseninin tekrari).
  2. **Sinif basina tek oturum.** 1 kosu / 1 firmware / 1 TCP yigini
     (lwIP) / 1 hedef / 1 port. Ayrilabilecek held-out YOK -> "saldiriyi
     mi ogrendi, bu ESP32 kosusunu mu" AYIRT EDILEMEZ (R35 riski).
  3. **Elimizdeki TEK gercek-dunya olcum varligi.** 2651 satir egitim
     icin kucuk (%0.4), degerlendirme icin tam dogru olcek. Egitime
     katmak, olcme yetenegimizi yok eder.

KARAR MANTIGI KOPYALANMAZ
-------------------------
`offline_test_multiclass.py::run_offline` OLDUGU GIBI cagrilir --
`validate_all_days_multiclass.py` ile ayni felsefe: "bir hatta port
edildi, digerinde unutuldu" sinifindan sapma uretilmez.

A/B KULLANIMI
-------------
Bayraklar ortam degiskeni oldugu icin ayni betik iki modda kosar:

    python analysis/validate_realworld_pcaps.py                  # uretim
    MC_R26_SOFTVOTE=1 python analysis/validate_realworld_pcaps.py  # soft-vote

Cikti:
    results/validate_realworld_pcaps_results.csv   (paket-bazli)
    results/validate_realworld_pcaps_summary.json  (ozet + baseline farki)

Kullanim (secmeli):
    SETS_FILTER="esp32_webattack_20260814" python analysis/validate_realworld_pcaps.py
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "multiclass_training"))

import pcap_validation_pilot as v1
import offline_test_multiclass as mc

OUT_CSV = os.path.join(PROJECT_DIR, "results", "validate_realworld_pcaps_results.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "results", "validate_realworld_pcaps_summary.json")

SETS_FILTER = os.environ.get("SETS_FILTER", "").strip()

ANALYZE_DIR = os.path.join(PROJECT_DIR, "data", "realworld")

# ---------------------------------------------------------------------------
# KAYIT DEFTERI (REGISTRY)
# ---------------------------------------------------------------------------
# Yeni bir gercek-dunya yakalamasi eklemek icin buraya bir girdi ekleyin.
# ZORUNLU alanlar: name, pcap, attacker_ip, expected_class.
# `baseline` = bu setin ILK olculdugu andaki referans rakamlar. Regresyonu
# gorunur kilar; YENI bir aday daha iyi cikarsa baseline GUNCELLENIR (ve
# hangi commit'te guncellendigi `baseline_source`'a yazilir).
#
# ONEMLI: `attack_gt` = "src_ip == attacker_ip". Bu, 2026-08-14 canli A/B
# olcumunun kullandigi KURALIN AYNISIDIR (saldirgan-kaynakli paketler);
# boylece asagidaki baseline rakamlari birebir yeniden uretilebilir.
# Yakalamanin geri kalani (ev/is trafigi) BENIGN arka plan sayilir ve
# ayri bir FP metrigi olarak raporlanir.
REALWORLD_SETS = [
    {
        "name": "esp32_webattack_20260814",
        "pcap": os.path.join(
            ANALYZE_DIR, "capture_esp32_webattack_softvote_20260814_1644.pcapng"),
        "attacker_ip": "192.168.1.103",
        "victim_ip": "192.168.1.101",
        "target_port": 3000.0,
        "expected_class": "WebAttack",
        "tool": "esp32_webattack.ino (SQLi/XSS/LFI, Connection: close)",
        "target_app": "OWASP Juice Shop :3000",
        "captured": "2026-08-14",
        "notes": (
            "FIZIKSEL olarak ayri cihaz (WSL kor-noktasi YOK). Kisa akis "
            "(istek basina yeni TCP baglantisi) -> R38 akis-sekli ekseni. "
            "R39: OOD DEGIL (ood_dist p50=3.69, esik 30, %0.00 asim); "
            "1-NN komsusu %99.1 BENIGN -> etiket-uzayi carpismasi."
        ),
        "baseline": {
            "baseline_source": "2026-08-14 kontrollu A/B (R26 Faz-2 Adim 2c)",
            "scored_attacker_packets": 2378,
            "production": {"binary_recall": 0.204, "behavioral_alerts": 376,
                           "volumetric_alerts": 108, "correct_class_rate": 0.0},
            "softvote":   {"binary_recall": 0.045, "behavioral_alerts": 0,
                           "volumetric_alerts": 108, "correct_class_rate": 0.0},
        },
    },
    {
        "name": "esp32_ftp_patator_20260814",
        "pcap": os.path.join(
            ANALYZE_DIR, "capture_esp32_ftp_softvote_20260814_1657.pcapng"),
        "attacker_ip": "192.168.1.103",
        "victim_ip": "192.168.1.101",
        "target_port": 2121.0,
        "expected_class": "FTP-Patator",
        "tool": "esp32_ftp_patator.ino (MODE_FAST)",
        "target_app": "FTP :2121",
        "captured": "2026-08-14",
        "notes": (
            "Hedef port 2121 EPHEMERAL kovaya dustu -> PortBucket_ftp=0, "
            "R24/K3 FTP-Patator yolu HIC calismadi. R39: 1-NN komsusu "
            "%100 BENIGN, FTP-Patator kumesi p50=8.96 ile en uzak "
            "ikinci sinif."
        ),
        "baseline": {
            "baseline_source": "2026-08-14 kontrollu A/B (R26 Faz-2 Adim 2d)",
            "scored_attacker_packets": 273,
            "production": {"binary_recall": 0.117, "behavioral_alerts": 32,
                           "volumetric_alerts": 0, "correct_class_rate": 0.0},
            "softvote":   {"binary_recall": 0.000, "behavioral_alerts": 0,
                           "volumetric_alerts": 0, "correct_class_rate": 0.0},
        },
    },
]


def packet_key(packet, i):
    """run_offline'in `packet_no` uretme kuralinin BIREBIR aynisi."""
    pn = packet.get("frame.number", i)
    return int(pn) if str(pn).isdigit() else i


def predicted_class(prediction_text, classes):
    """`prediction` serbest metin; icinden GERCEK sinif adini cikar.

    DIKKAT: R37/D1'in port-farkindalikli adi parantez icinde ACIKLAMA
    tasiyabiliyor, orn.
        "Hacimsel/SYN-Flood -> DoS (RF-override, port=3000: uygulama-katmani
         seli -- DoS/WebAttack-flood ayrimi BELIRSIZ)"
    Duz `substring` aramasi bu aciklamadaki "WebAttack"i TAHMIN sanir
    (ilk yazimda tam bu hata yapildi ve 90 DoS alarmi yanlislikla
    "WebAttack" sayildi). Bu yuzden once parantezli aciklama ATILIR,
    sonra varsa "->"nin SAGINDAKI parca kullanilir.
    """
    if not prediction_text:
        return None
    head = str(prediction_text).split(" (")[0]
    if "->" in head:
        head = head.rsplit("->", 1)[1]
    hits = [c for c in classes if c in head]
    if not hits:
        return None
    return max(hits, key=len)  # en uzun eslesme (DDoS vs DoS)


def main():
    softvote_on = os.environ.get("MC_R26_SOFTVOTE", "0") != "0"
    mode = "softvote" if softvote_on else "production"

    print("=" * 78)
    print("GERCEK-DUNYA HELD-OUT DEGERLENDIRMESI  (TEST-ONLY -- EGITIME KATILMAZ)")
    print(f"mod: {mode}   (MC_R26_SOFTVOTE={os.environ.get('MC_R26_SOFTVOTE','0')})")
    print("=" * 78)

    rf, lstm, pipeline, classes, n_classes, benign_idx = mc.load_multiclass_models()
    classes = list(classes)

    sets = REALWORLD_SETS
    if SETS_FILTER:
        wanted = set(SETS_FILTER.split(","))
        sets = [s for s in REALWORLD_SETS if s["name"] in wanted]
        print(f"[*] SETS_FILTER aktif -> {len(sets)} set")

    all_rows = []
    summary = {"mode": mode, "sets": []}

    for spec in sets:
        pcap = spec["pcap"]
        if not os.path.exists(pcap):
            print(f"[!] BULUNAMADI, atlaniyor: {pcap}")
            continue

        print(f"\n[{spec['name']}] {os.path.basename(pcap)}", flush=True)
        packets = v1.read_pcap(pcap)
        print(f"    {len(packets)} kare okundu, multiclass dispatcher calisiyor...",
              flush=True)

        pkt_by_key = {}
        for i, p in enumerate(packets):
            pkt_by_key.setdefault(packet_key(p, i), p)

        results = mc.run_offline(rf, lstm, pipeline, classes,
                                 n_classes, benign_idx, packets)

        atk = {"n": 0, "alert": 0, "behav": 0, "vol": 0,
               "correct_class": 0, "ood": 0}
        bg = {"n": 0, "alert": 0}
        pred_counter = {}

        for r in results:
            src_pkt = pkt_by_key.get(r["packet_no"])
            if src_pkt is None:
                continue
            is_attacker = (r["src_ip"] == spec["attacker_ip"])
            # canli hattaki status onceligi: UNKNOWN > ATTACK > BENIGN
            alert = bool(r["is_alert"] and not r["is_ood"])

            if is_attacker:
                atk["n"] += 1
                if r["is_ood"]:
                    atk["ood"] += 1
                if alert:
                    atk["alert"] += 1
                    if r["is_volumetric"]:
                        atk["vol"] += 1
                    else:
                        atk["behav"] += 1
                    pc = predicted_class(r.get("prediction"), classes)
                    pred_counter[pc] = pred_counter.get(pc, 0) + 1
                    if pc == spec["expected_class"]:
                        atk["correct_class"] += 1
            else:
                bg["n"] += 1
                if alert:
                    bg["alert"] += 1

            all_rows.append({
                "set": spec["name"], "mode": mode,
                "packet_no": r["packet_no"], "src_ip": r["src_ip"],
                "dst_ip": r["dst_ip"], "dst_port": r["dst_port"],
                "is_attacker": is_attacker,
                "is_volumetric": r["is_volumetric"], "risk_score": r["risk_score"],
                "prediction": r["prediction"], "is_alert": r["is_alert"],
                "is_ood": r["is_ood"], "ood_distance": r["ood_distance"],
            })

        recall = atk["alert"] / atk["n"] if atk["n"] else 0.0
        ccr = atk["correct_class"] / atk["n"] if atk["n"] else 0.0
        bg_far = bg["alert"] / bg["n"] if bg["n"] else 0.0

        base = spec["baseline"].get(mode, {})
        d_recall = recall - base.get("binary_recall", float("nan"))

        print(f"    saldirgan paketi (skorlanan) : {atk['n']}"
              f"   [baseline {spec['baseline'].get('scored_attacker_packets')}]")
        print(f"    IKILI RECALL                 : {recall:.4f}"
              f"   [baseline {base.get('binary_recall')}]  delta={d_recall:+.4f}")
        print(f"      davranissal dal alarmi     : {atk['behav']}"
              f"   [baseline {base.get('behavioral_alerts')}]")
        print(f"      hacimsel dal alarmi        : {atk['vol']}"
              f"   [baseline {base.get('volumetric_alerts')}]")
        print(f"    DOGRU SINIF ('{spec['expected_class']}') orani : {ccr:.4f}"
              f"   [baseline {base.get('correct_class_rate')}]")
        print(f"    saldirgan paketlerinde UNKNOWN/OOD : {atk['ood']}")
        print(f"    alarm veren siniflar         : {pred_counter}")
        print(f"    arka plan (ev/is trafigi)    : n={bg['n']}, "
              f"FP={bg['alert']}, FAR={bg_far:.5f}")

        summary["sets"].append({
            "name": spec["name"], "expected_class": spec["expected_class"],
            "tool": spec["tool"], "target_app": spec["target_app"],
            "captured": spec["captured"], "notes": spec["notes"],
            "attacker_packets_scored": atk["n"],
            "binary_recall": round(recall, 4),
            "behavioral_alerts": atk["behav"], "volumetric_alerts": atk["vol"],
            "correct_class_rate": round(ccr, 4),
            "attacker_unknown_ood": atk["ood"],
            "predicted_classes": pred_counter,
            "background_packets": bg["n"], "background_fp": bg["alert"],
            "background_far": round(bg_far, 5),
            "baseline": base,
            "delta_binary_recall": round(d_recall, 4),
        })

    os.makedirs(os.path.join(PROJECT_DIR, "results"), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("HATIRLATMA: bu yakalamalar TEST-ONLY'dir; egitim CSV'sine "
          "EKLENMEZ (gerekce dosya basligindaki 3 madde).")
    print(f"[+] {OUT_CSV}")
    print(f"[+] {OUT_JSON}")


if __name__ == "__main__":
    main()
