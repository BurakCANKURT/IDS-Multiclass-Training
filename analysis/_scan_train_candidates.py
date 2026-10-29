# -*- coding: utf-8 -*-
"""
_scan_train_candidates.py
==========================
validate_all_days.py'nin GROUPS validasyon batch'leriyle CAKISMAYAN (disjoint)
Tuesday/Thursday/Friday batch indekslerini tarar; her aday batch icin ilgili
gt_func ile gt_attack=1 paket sayisi + toplam paket sayisini raporlar.

Cikti: _scan_train_candidates.json
  {family: [{idx, total, attack, pcap}, ...]}  (attack>0 olanlar, attack desc sirali)
"""
import os
import json
import glob
import warnings
warnings.filterwarnings("ignore")

import pcap_validation_pilot as v1
from validate_all_days import BATCH_DIRS, find_batch_day, GROUPS, make_pair_port_gt, gt_bot

PROJECT_DIR = v1.PROJECT_DIR
OUT_JSON = os.path.join(PROJECT_DIR, "results", "_scan_train_candidates.json")

# (day, family, gt_func, validation_batches, scan_range(start,end,step))
SCAN_DEFS = [
    ("Tuesday", "FTP-Patator",
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {21}, "FTP-Patator"),
        ["00570", "00630", "00650", "00660"], (520, 700, 2)),
    ("Tuesday", "SSH-Patator",
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {22}, "SSH-Patator"),
        ["00930", "00950", "00990", "01020"], (870, 1080, 2)),
    ("Thursday", "WebAttack",
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {80}, "WebAttack"),
        ["00300", "00350", "00360", "00370"], (260, 450, 2)),
    ("Thursday", "Infiltration",
        make_pair_port_gt("192.168.10.8", "205.174.165.73", {444}, "Infiltration"),
        ["00690", "00700"], (550, 933, 3)),
    ("Friday", "Bot",
        gt_bot,
        ["00460", "00470"], (350, 600, 2)),
    ("Friday", "PortScan",
        make_pair_port_gt("172.16.0.1", "192.168.10.50", None, "PortScan"),
        ["00700", "00720", "00750"], (650, 850, 2)),
    ("Friday", "DDoS",
        make_pair_port_gt("172.16.0.1", "192.168.10.50", {80}, "DDoS"),
        ["00830", "00870", "00920"], (790, 1000, 2)),
]


def main():
    results = {}
    for day, family, gt_func, val_batches, (lo, hi, step) in SCAN_DEFS:
        print(f"\n=== {day}/{family}  scan range {lo}-{hi} step {step} (val={val_batches}) ===", flush=True)
        rows = []
        for i in range(lo, hi, step):
            idx = f"{i:05d}"
            if idx in val_batches:
                continue
            pcap = find_batch_day(day, idx)
            if not pcap:
                continue
            try:
                packets = v1.read_pcap(pcap)
            except Exception as e:
                print(f"  [!] {idx}: read error {e}")
                continue
            n_attack = 0
            for pkt in packets:
                gt, _ = gt_func(pkt)
                n_attack += gt
            if n_attack > 0:
                rows.append({"idx": idx, "total": len(packets), "attack": n_attack,
                              "pcap": os.path.basename(pcap)})
                print(f"  {idx}: total={len(packets):5d} attack={n_attack:5d}", flush=True)
        rows.sort(key=lambda r: -r["attack"])
        results[f"{day}/{family}"] = rows

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] saved {OUT_JSON}")


if __name__ == "__main__":
    main()
