# -*- coding: utf-8 -*-
"""
validate_all_days_multiclass.py
================================
`validate_all_days.py`'nin MULTICLASS karsiligi: MulticlassRF/MulticlassLSTM
hattini (MC1 kapsami, 7 sinif) CICIDS-2017'nin capraz-gun (Tuesday/Thursday/
Friday) batch'lerinde calistirir ve ikili hatla AYNI protokolde
DR/FAR/precision/MCC uretir.

NEDEN VAR: ikili hattin capraz-gun referans rakamlari (R28: MCC 0.9633 /
FAR 0.0069 / DR 0.9670) `validate_all_days.py`'den geliyordu; multiclass
tarafinda ESDEGER bir harness YOKTU. "Multiclass ana hat olsun mu" karari
kiyaslanabilir rakam olmadan verilemez -- bu betik o boslugu kapatir.

## IKI KRITIK OLCUM-HIJYENI KARARI

1) CIFT-YONLU GROUND TRUTH (B5 duzeltmesi).
   `offline_test_multiclass.py`'nin GT'si TEK YONLUYDU
   (`src==172.16.0.1 AND dst==192.168.10.50 AND dst_port==80`), bu yuzden
   sistemin DOGRU yakaladigi ters-yon saldiri paketleri FP sayiliyor ve
   multiclass FAR'i ~31x sisiyordu (olculdu: tek-yonlu FAR=0.0434 vs
   cift-yonlu FAR=0.0014, ayni CSV/ayni kosu). Bu betik ikili harness'in
   `make_pair_port_gt`/`gt_bot` fonksiyonlarini AYNEN kullanir (pair
   eslesmesi = {src,dst} kumesi, port her iki yonde de kabul) -> iki hattin
   FAR'lari artik DOGRUDAN kiyaslanabilir.

2) MC1 KAPSAMI ayri raporlanir.
   MC1 sinif kumesini 9'dan 7'ye indirdi: **Bot ve Infiltration kapsam
   disi**. Model bu iki sinifi URETEMEZ, dolayisiyla onlari ana metriğe
   katmak modeli emitleyemeyecegi bir etiketten sorumlu tutmak olur.
   Ana rakam `overall_in_scope` (Bot/Infiltration gruplari HARIC);
   `overall_all` seffaflik icin yine de hesaplanir. Ikili hattin
   `validate_all_days.py` rakami TUM gruplar uzerindendir -- birebir
   kiyas icin `overall_all` satirini kullan, kapsam-adil kiyas icin
   `overall_in_scope`.

## IKILI KARARA COKERTME (binary collapse)
Multiclass hat zaten ATTACK/BENIGN/UNKNOWN uretiyor; ikili sinyal
`argmax != BENIGN` degil, canli hattaki ile AYNI kural:
    alert = (risk_score >= ALERT_THRESHOLD) AND (NOT is_ood)
cunku `process_pcap_multiclass.py`'de status onceligi UNKNOWN > ATTACK'tir
(R20 kapisi). Seffaflik icin UNKNOWN'i alarm SAYAN alternatif metrik de
(`*_ood_as_alert`) raporlanir.

DIKKAT: bu betik hicbir esik/kapi/veto DEGISTIRMEZ. Multiclass karar
mantigi `offline_test_multiclass.py::run_offline`'dan OLDUGU GIBI cagrilir
(kod kopyalanmaz) -- boylece "bir hatta port edildi digerinde unutuldu"
sinifindan yeni bir sapma yaratilmaz.

Cikti:
    results/validate_all_days_multiclass_results.csv   (paket-bazli)
    results/validate_all_days_multiclass_summary.json  (ozet metrikler)

Kullanim:
    python analysis/validate_all_days_multiclass.py
    GROUPS_FILTER="Friday/DDoS,Tuesday/Clean" python analysis/validate_all_days_multiclass.py
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "multiclass_training"))

import pcap_validation_pilot as v1
import validate_all_days as vad
import offline_test_multiclass as mc

OUT_CSV = os.path.join(PROJECT_DIR, "results", "validate_all_days_multiclass_results.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "results", "validate_all_days_multiclass_summary.json")

GROUPS_FILTER = os.environ.get("GROUPS_FILTER", "").strip()

# MC1 kapsami disinda kalan gruplar -- model bu siniflari uretemez.
OUT_OF_SCOPE_GROUPS = {"Bot", "Infiltration"}


def packet_key(packet, i):
    """run_offline'in `packet_no` uretme kuralinin BIREBIR aynisi -- sonuc
    satirlarini kaynak pakete geri eslemek icin (GT paketten hesaplanir)."""
    pn = packet.get("frame.number", i)
    return int(pn) if str(pn).isdigit() else i


def main():
    rf, lstm, pipeline, classes, n_classes, benign_idx = mc.load_multiclass_models()

    groups = vad.GROUPS
    if GROUPS_FILTER:
        wanted = set(GROUPS_FILTER.split(","))
        groups = [g for g in vad.GROUPS if f"{g[0]}/{g[1]}" in wanted]
        print(f"[*] GROUPS_FILTER aktif: {sorted(wanted)} -> {len(groups)} grup calisacak")

    all_rows = []
    batch_summ = []

    for day, family, batches, gt_func in groups:
        for idx in batches:
            pcap = vad.find_batch_day(day, idx)
            if not pcap:
                print(f"[!] {day} batch {idx} bulunamadi, atlaniyor.")
                continue
            print(f"\n[{day} | grup={family} | batch {idx}] {os.path.basename(pcap)}", flush=True)
            packets = v1.read_pcap(pcap)
            print(f"    {len(packets)} paket okundu, multiclass dispatcher calisiyor...", flush=True)

            # Paketleri run_offline'in kullandigi anahtarla indeksle: sonuc
            # satirlari (IP'siz/hatali paketler atlandigi icin) girdiyle
            # 1:1 hizalanmaz, bu yuzden zip DEGIL anahtar-eslemesi sart.
            pkt_by_key = {}
            for i, p in enumerate(packets):
                pkt_by_key.setdefault(packet_key(p, i), p)

            # Multiclass karar mantigi burada YENIDEN YAZILMAZ; uretim
            # harness'i oldugu gibi cagrilir.
            results = mc.run_offline(rf, lstm, pipeline, classes, n_classes, benign_idx, packets)

            y_true, y_pred = [], []
            n_rf = n_lstm = 0
            for r in results:
                src_pkt = pkt_by_key.get(r["packet_no"])
                if src_pkt is None:
                    continue
                gt, fam = gt_func(src_pkt)
                # Canli hattaki status onceligi: UNKNOWN > ATTACK > BENIGN.
                alert = 1 if (r["is_alert"] and not r["is_ood"]) else 0
                alert_ood_as_alert = 1 if r["is_alert"] else 0
                y_true.append(gt)
                y_pred.append(alert)
                if r["is_volumetric"]:
                    n_rf += 1
                else:
                    n_lstm += 1
                all_rows.append({
                    "day": day, "group": family, "batch": idx,
                    "packet_no": r["packet_no"],
                    "src_ip": r["src_ip"], "dst_ip": r["dst_ip"], "dst_port": r["dst_port"],
                    "model": r["model"], "is_volumetric": r["is_volumetric"],
                    "risk_score": r["risk_score"], "counter_score": r["counter_score"],
                    "prediction": r["prediction"],
                    "is_ood": r["is_ood"], "ood_distance": r["ood_distance"],
                    "ood_pvalue": r["ood_pvalue"],
                    "is_committee_veto": r["is_committee_veto"],
                    # (GATE-SC-0, 2026-08-31) salt-gozlem yapisal sayaclar --
                    # karar mantigi DEGISMEDI, bkz. offline_test_multiclass.py.
                    "sc_vol_syn": r.get("sc_vol_syn"),
                    "sc_vol_ratio": r.get("sc_vol_ratio"),
                    "sc_distinct_ports": r.get("sc_distinct_ports"),
                    "sc_distinct_conns": r.get("sc_distinct_conns"),
                    "gt_attack": gt, "gt_family": fam,
                    "alert": alert, "alert_ood_as_alert": alert_ood_as_alert,
                    "outcome": vad.tp_fp_fn_tn_label(gt, alert),
                    "in_scope": family not in OUT_OF_SCOPE_GROUPS,
                })

            tp, fp, tn, fn = vad.confusion(y_true, y_pred)
            bs = {
                "day": day, "group": family, "batch": idx, "pcap": os.path.basename(pcap),
                "total_pkts": len(packets), "scored_pkts": len(y_true),
                "gt_attack_pkts": int(sum(y_true)),
                "gt_benign_pkts": int(len(y_true) - sum(y_true)),
                "alerts": int(sum(y_pred)),
                "n_RF": n_rf, "n_LSTM": n_lstm,
            }
            bs.update(vad.metrics_from_conf(tp, fp, tn, fn))
            batch_summ.append(bs)
            print(f"    pkts={len(packets)} scored={len(y_true)} gt_attack={sum(y_true)} "
                  f"alerts={sum(y_pred)} RF={n_rf} LSTM={n_lstm} | "
                  f"DR={bs['DR_recall_attack']} FAR={bs['FAR']} MCC={bs['MCC']}", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "models": "multiclass (MC1, 7 sinif)",
        "alert_threshold": mc.ALERT_THRESHOLD,
        "ood_threshold": mc.OOD_DISTANCE_THRESHOLD,
        "gt_definition": "bidirectional pair-match (validate_all_days.py ile ayni) -- B5 tek-yonlu GT hatasi duzeltildi",
        "binary_collapse_rule": "alert = risk>=ALERT_THRESHOLD AND NOT is_ood",
        "out_of_scope_groups": sorted(OUT_OF_SCOPE_GROUPS),
        "batches": batch_summ,
    }

    per_group = {}
    for day, family, _, _ in vad.GROUPS:
        key = f"{day}/{family}"
        if key in per_group:
            continue
        sub = df[(df["day"] == day) & (df["group"] == family)]
        if sub.empty:
            continue
        tp, fp, tn, fn = vad.confusion(sub["gt_attack"], sub["alert"])
        m = vad.metrics_from_conf(tp, fp, tn, fn)
        m["in_scope"] = family not in OUT_OF_SCOPE_GROUPS
        m["dispatch_vs_gt"] = (
            sub.groupby(["gt_family", "prediction"]).size()
            .unstack(fill_value=0).to_dict(orient="index")
        )
        per_group[key] = m
    summary["per_group"] = per_group

    def block(frame, label):
        if frame.empty:
            return None
        tp, fp, tn, fn = vad.confusion(frame["gt_attack"], frame["alert"])
        out = vad.metrics_from_conf(tp, fp, tn, fn)
        tp2, fp2, tn2, fn2 = vad.confusion(frame["gt_attack"], frame["alert_ood_as_alert"])
        out["variant_ood_as_alert"] = vad.metrics_from_conf(tp2, fp2, tn2, fn2)
        out["label"] = label
        return out

    in_scope = df[df["in_scope"]]
    summary["overall_in_scope"] = block(in_scope, "MC1 kapsami (Bot/Infiltration haric)")
    summary["overall_all"] = block(df, "TUM gruplar (ikili validate_all_days.py ile birebir kiyas)")
    summary["overall_attack_groups_in_scope"] = block(
        in_scope[in_scope["group"] != "Clean"], "MC1 kapsami, Clean haric")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print("GUN / GRUP BAZLI SONUC (multiclass):")
    for key, m in per_group.items():
        flag = "" if m["in_scope"] else "  [MC1 KAPSAM DISI]"
        print(f"  {key:24s}: DR={m['DR_recall_attack']} FAR={m['FAR']} "
              f"precision={m['precision_attack']} MCC={m['MCC']} "
              f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}{flag}")
    for k in ("overall_in_scope", "overall_all"):
        if summary.get(k):
            m = summary[k]
            print(f"\n{k} ({m['label']}):")
            print(f"   DR={m['DR_recall_attack']} FAR={m['FAR']} precision={m['precision_attack']} "
                  f"MCC={m['MCC']} TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")
    print(f"\n[+] Paket-bazli sonuc : {OUT_CSV}")
    print(f"[+] Ozet (JSON)       : {OUT_JSON}")


if __name__ == "__main__":
    main()
