# -*- coding: utf-8 -*-
"""
build_streaming_dataset.py
===========================
pcap_validation_pilot.py'nin DR=0.0719/FAR=0.0397/MCC=0.0708 sonucunun kok
nedeni: MainRF/MainLSTM, CICFlowMeter'in TAMAMLANMIS akis agregalariyla
egitildi, ama UniversalRealTimeFlowTable.extract_universal_features() paket
paket / akis-devam-ederken hesaplanan INCREMENTAL ozellik vektorleri uretiyor.

Bu betik, AYNI incremental ozellik cikarimini (pcap_validation_pilot.py'den
1:1 import edilerek) 28 Wednesday batch'ine uygulayip 32-ozellik + binary
label (ATTACK=172.16.0.1<->192.168.10.50:80) iceren bir egitim seti uretir.

Secilen 28 batch, pcap_validation_pilot.py'nin 12 dogrulama batch'i
(00700,00750,00800,00850,00900,01010,00100,00200,00300,00400,00500,01100)
ile KESISMEZ -> train/validation sizintisi yok.

Cikti: streaming_training_dataset.csv (32 ozellik + label + batch)
"""

import os
import csv
import joblib
import pandas as pd

from pcap_validation_pilot import (
    UniversalRealTimeFlowTable, gt_label, read_pcap, find_batch,
    PROJECT_DIR, BUNDLE_PATH,
)

TRAIN_BATCHES = [
    "00020", "00060", "00080", "00150", "00250", "00350", "00450", "00550",
    "00620", "00660", "00680", "00720", "00760", "00780", "00820", "00860",
    "00880", "00920", "00940", "00960", "00980", "01000", "01020",
    "01150", "01200", "01250", "01300", "01350",
]

VALIDATION_BATCHES = {
    "00700", "00750", "00800", "00850", "00900", "01010",
    "00100", "00200", "00300", "00400", "00500", "01100",
}

OUT_CSV = os.path.join(PROJECT_DIR, "data", "streaming_training_dataset.csv")


def main():
    assert not (set(TRAIN_BATCHES) & VALIDATION_BATCHES), "train/validation overlap!"

    bundle = joblib.load(BUNDLE_PATH)
    raw_columns = bundle["raw_columns"]

    rows = []
    for idx in TRAIN_BATCHES:
        pcap = find_batch(idx)
        if not pcap:
            print(f"[!] batch {idx} bulunamadi, atlaniyor.")
            continue
        print(f"[batch {idx}] {os.path.basename(pcap)}", flush=True)
        packets = read_pcap(pcap)
        print(f"    {len(packets)} paket okundu, ozellik cikarimi...", flush=True)

        flow_table = UniversalRealTimeFlowTable()
        n_attack = 0
        for pkt in packets:
            master_features, _ = flow_table.extract_universal_features(pkt)
            if master_features is None:
                continue
            label = gt_label(pkt)
            n_attack += label
            row = {col: float(master_features.get(col, 0.0)) for col in raw_columns}
            row["label"] = label
            row["batch"] = idx
            rows.append(row)
        print(f"    -> {len(packets)} pkt, {n_attack} attack-labeled", flush=True)

    df = pd.DataFrame(rows, columns=raw_columns + ["label", "batch"])
    df.to_csv(OUT_CSV, index=False)
    print("\n" + "=" * 60)
    print(f"Toplam satir: {len(df)}")
    print(f"Label dagilimi:\n{df['label'].value_counts()}")
    print(f"[+] Kaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
