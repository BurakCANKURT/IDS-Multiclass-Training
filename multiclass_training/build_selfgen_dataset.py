# -*- coding: utf-8 -*-
"""
build_selfgen_dataset.py
=========================
Kendi (self-generation, WSL2 + native loopback hedef) canli-ates kampanyasindan
toplanan pcap'leri egitim satirlarina cevirir.

ONEMLI -- IP KULLANIMI HAKKINDA
--------------------------------
IP adresi (127.0.0.1) modele HICBIR SEKILDE ozellik olarak verilmez -- RAW_COLUMNS_V5
(41 sutun, pcap_validation_pilot_v5.py) zaten ip.src/ip.dst icermez, sadece
PortBucket_* + paket-uzunlugu/zamanlama/flag istatistikleri. IP, gt_label() ile
AYNI ESKI/ONAYLANMIS deseni izleyerek SADECE ground-truth ETIKETINI kurarken
kullanilir (CICIDS'in kendi orijinal etiketlemesiyle ayni felsefe).

Loopback'te saldirgan==hedef IP oldugu icin (WSL2 mirrored -> 127.0.0.1) IP ile
yon ayirt edilemez -- bunun yerine PORT + ZAMAN PENCERESI (campaign_run.sh'in
urettigi phases.json) kullanilir: bir paketin zamani [start,end) araligina VE
dst_port'u o fazin portuna denk geliyorsa, o fazin label'i atanir; aksi halde
"BENIGN" (arka plan / eslesmeyen trafik).

Cikti: multiclass_training/data/selfgen_dataset.csv (production dataset'in
UZERINE YAZMAZ -- ayri dosya, once incelenip sonra elle birlestirilir).

Kullanim:
    python build_selfgen_dataset.py <pcap_path> <phases_json_path> [batch_tag]
"""
import os
import sys
import json
import csv

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "analysis"))

import pcap_validation_pilot as v1
from pcap_validation_pilot_v5 import UniversalRealTimeFlowTableV5, RAW_COLUMNS_V5

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "selfgen_dataset.csv")
CLASSES_JSON = os.environ.get(
    "MC_CLASSES_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "streaming_label_classes.json"))


def load_class_order():
    # 19. KURAL (R49): iki CSV'nin sema uyumu deger SEMANTIGINI garanti etmez.
    # `label` uretim CSV'sinde sorted(classes).index(label_str) ile kodlanir
    # (BENIGN=0, DDoS=1, DoS=2, FTP-Patator=3, PortScan=4, SSH-Patator=5,
    # WebAttack=6) -- bu ikili (0/1) bir kodlama DEGIL. Once burada da HARDCODE
    # ikili idi (R49'da bulunup R49-v2'de duzeltildi): ham pd.concat + df["label"]
    # ile egitilirse WebAttack/FTP-Patator satirlari sessizce "DDoS" (index 1)
    # olurdu.
    with open(CLASSES_JSON, encoding="utf-8") as f:
        classes = json.load(f)
    return sorted(classes)


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def load_phases(phases_json_path):
    with open(phases_json_path, "r", encoding="utf-8") as f:
        phases = json.load(f)
    # start/end WSL tarafinda uretildi (epoch saniye) -- pcap'teki frame.time_epoch
    # ile AYNI saat kaynagi degilse (WSL vs Windows host clock skew) sistematik
    # kaymaya yol acar. Ikisi de ayni fiziksel makinede (WSL2 host saatini
    # PAYLASIR, ayri saat degil) oldugundan skew pratikte ~0 olmali.
    return phases


def find_phase(ts, dst_port, phases):
    for ph in phases:
        if ph["dst_port"] == dst_port and ph["start"] <= ts <= ph["end"]:
            return ph
    return None


def main():
    if len(sys.argv) < 3:
        print("Kullanim: python build_selfgen_dataset.py <pcap> <phases_json> [batch_tag]")
        sys.exit(1)
    pcap_path = sys.argv[1]
    phases_path = sys.argv[2]
    batch_tag = sys.argv[3] if len(sys.argv) > 3 else "selfgen_01"

    class_order = load_class_order()
    phases = load_phases(phases_path)
    print(f"[*] {len(phases)} faz yuklendi:")
    for ph in phases:
        dur = ph["end"] - ph["start"]
        print(f"    {ph['tool']:24s} port={ph['dst_port']:<5} label={ph['label']:12s} dur={dur:.1f}s")

    print(f"[*] pcap okunuyor: {pcap_path}")
    packets = v1.read_pcap(pcap_path)
    print(f"    {len(packets)} kare okundu")

    # Zaman sirasina gore isle -- flow tablosu streaming/stateful.
    def pkt_ts(p):
        return _to_float(p.get("frame.time_epoch"), 0.0)
    packets.sort(key=pkt_ts)

    flow_table = UniversalRealTimeFlowTableV5()

    # Hizmet portlari HARDCODE edilmez -- phases.json'da fiilen kullanilan
    # tum dst_port degerlerinden turetilir (80/21 VE 3000/2121 ephemeral-bucket
    # varyantlari dahil). Ilk turda bu sabitlenmisti (80,21) ve 3000/2121
    # dongulerindeki TUM satirlar sessizce ATLANMISTI -- duzeltildi.
    known_service_ports = {int(ph["dst_port"]) for ph in phases}
    print(f"[*] bilinen hizmet portlari (phases.json'dan): {sorted(known_service_ports)}")

    rows = []
    label_counts = {}
    for p in packets:
        ts = pkt_ts(p)
        dport = _to_int(p.get("tcp.dstport"), 0)
        sport = _to_int(p.get("tcp.srcport"), 0)
        proto = p.get("_ws.col.Protocol", "")
        if not (p.get("tcp.srcport") or p.get("tcp.dstport")):
            continue  # TCP-disi kare, flow ozellikleri tanimsiz

        # Yon: dst_port bilinen bir hizmet portuysa bu FORWARD (istemci->sunucu)
        # paket -- faz araniyor. Aksi yon (sunucu->istemci yaniti) AYNI akisin
        # parcasi, ayri etiketlenmez (CICIDS'te de akis TEK etiket alir).
        svc_port = dport if dport in known_service_ports else (sport if sport in known_service_ports else None)
        if svc_port is None:
            continue

        phase = find_phase(ts, svc_port, phases)
        label_str = phase["label"] if phase else "BENIGN"

        master_features, flow_key = flow_table.extract_universal_features(p)
        if master_features is None:
            continue

        row = [master_features[col] for col in RAW_COLUMNS_V5]
        label = class_order.index(label_str)
        row += [label, label_str, "SelfGen", label_str if label_str != "BENIGN" else "Benign-SelfGen", batch_tag]
        rows.append(row)
        label_counts[label_str] = label_counts.get(label_str, 0) + 1

    print(f"[*] {len(rows)} satir uretildi. Etiket dagilimi:")
    for k, v in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:20s} {v}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(RAW_COLUMNS_V5 + ["label", "label_str", "day", "family", "batch"])
        w.writerows(rows)
    print(f"[+] yazildi: {OUT_CSV} (append modu, write_header={write_header})")


if __name__ == "__main__":
    main()
