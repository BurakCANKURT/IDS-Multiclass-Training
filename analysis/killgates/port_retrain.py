# -*- coding: utf-8 -*-
"""
Port-kovasi DUSURULMUS RF+HGB retrain, >=3 tohum (hakem itirazi DG M3).

DG 4.8'in "port ablasyonu" kaniti bir CIKARIM-ANI mudahalesiydi (R24:
PortBucket_* egitilmis modelde BENIGN medyanina cekildi -> FTP/SSH-Patator
recall 1.0 -> 0.0). Bu BAGIMLILIGI gosterir, ZORUNLULUGU degil (36b. KURAL:
R51'de -99 puanlik donmus-model cokusu retrain'de -0.2'ye inmisti). Bu betik
gercek retrain'i kosar.

Mudahale: 7 PortBucket_* sutunu OLCEKLENMIS uzayda 0.0'a sabitlenir (sutun
sayisi 41 kalir; R51 ile ayni desen). RF (uretim hiperparametreleri) ve HGB
(uretim hiperparametreleri), her biri 3 tohum; ayni kosulda port-DAHIL kontrol
kollari da ayni tohumlarla egitilir (tohum etkisi ile mudahale etkisi ayrissin).

Degerlendirme -- MODEL duzeyi (36a. KURAL: mudahalenin hedefledigi bilesenin
kendi argmax'i; uctan uca hat LSTM'i de toplar):
  * korpus test bolmesi: ikili recall, FPR, sinif-dogru recall (FTP/SSH-Patator)
  * ND Tablo 1'in FTP/SSH held-out batch'leri (validate_all_days.GROUPS)
  * ESP32 FTP-Patator ve WebAttack yakalamalari: saldirgan satirlarinda
    argmax != BENIGN ve argmax == dogru sinif; arka plan satirlarinda
    argmax != BENIGN (FP vekili)
Hicbir dosya uretime yazilmaz; modeller bellekte kalir.

Kullanim: python analysis/killgates/port_retrain.py [--seeds 3]
Cikti   : analysis/killgates/out/port_retrain.json
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAMS = ["FTP-Patator", "SSH-Patator"]


def heldout_rows():
    import validate_all_days as vad
    import build_multiclass_streaming_dataset as bm
    rows = []
    for day, fam, batches, gt in vad.GROUPS:
        if fam not in FAMS:
            continue
        for idx in batches:
            with contextlib.redirect_stdout(io.StringIO()):
                rows += bm._extract_multiclass(vad.find_batch_day(day, idx), gt, fam, day, idx)
    return pd.DataFrame(rows)


def esp32_rows():
    import pcap_validation_pilot as v1
    import validate_realworld_pcaps as vr
    with contextlib.redirect_stdout(io.StringIO()):
        import offline_test_multiclass as mc
    out = {}
    for spec in vr.REALWORLD_SETS:
        ft = mc.UniversalRealTimeFlowTableV5()
        rows, atk = [], []
        for p in v1.read_pcap(spec["pcap"]):
            if mc._is_non_ip(p):
                continue
            mf, _ = ft.extract_universal_features(p)
            if mf is None:
                continue
            rows.append([float(mf.get(c, 0.0)) for c in common.raw_columns()])
            atk.append(p.get("ip.src") == spec["attacker_ip"])
        out[spec["name"]] = (pd.DataFrame(rows, columns=common.raw_columns()),
                             np.array(atk), spec["expected_class"])
    return out


def evaluate(model, Xs_by_set, classes, drop_idx):
    b = classes.index("BENIGN")
    res = {}
    for name, (X, y_str, extra) in Xs_by_set.items():
        X = X.copy()
        if drop_idx:
            X[:, drop_idx] = 0.0
        pred = model.predict_proba(X).argmax(1)
        if name == "corpus_test":
            atk = y_str != "BENIGN"
            res[name] = {"binary_recall": float((pred[atk] != b).mean()),
                         "fpr": float((pred[~atk] != b).mean())}
            for f in FAMS:
                m = y_str == f
                res[name][f"{f}_class_recall"] = float((pred[m] == classes.index(f)).mean())
        elif name == "table1_heldout":
            res[name] = {}
            for f in FAMS:
                m = y_str == f
                res[name][f"{f}_class_recall"] = float((pred[m] == classes.index(f)).mean())
                res[name][f"{f}_binary_recall"] = float((pred[m] != b).mean())
            res[name]["benign_fpr"] = float((pred[y_str == "BENIGN"] != b).mean())
        else:
            is_atk, expected = y_str, extra
            res[name] = {
                "attacker_binary": float((pred[is_atk] != b).mean()),
                "attacker_correct_class": float((pred[is_atk] == classes.index(expected)).mean()),
                "background_alarm": float((pred[~is_atk] != b).mean()),
                "attacker_rows": int(is_atk.sum()), "background_rows": int((~is_atk).sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    cols = common.raw_columns()
    drop_idx = [cols.index(c) for c in cols if c.startswith("PortBucket_")]
    assert len(drop_idx) == 7
    tr, te, classes = common.load_corpus_split()
    rf_p, hgb_p, _, pipeline, _ = common.load_models("")
    T = lambda df: np.nan_to_num(pipeline.transform(df[cols]))  # noqa: E731

    ho = heldout_rows()
    esp = esp32_rows()
    sets = {"corpus_test": (T(te), te["label_str"].values, None),
            "table1_heldout": (T(ho), ho["label_str"].values, None)}
    for name, (X, atk, exp) in esp.items():
        sets[name] = (np.nan_to_num(pipeline.transform(X)), atk, exp)
    print(f"held-out satir={len(ho)}; esp32 setleri={list(esp)}", flush=True)

    Xtr, ytr = T(tr), tr["label"].values.astype(int)
    Xtr_drop = Xtr.copy()
    Xtr_drop[:, drop_idx] = 0.0

    result = {"production": {"rf": evaluate(rf_p, sets, classes, []),
                             "hgb": evaluate(hgb_p, sets, classes, [])},
              "arms": {}}
    for s in range(a.seeds):
        for arm, X, di in (("with_port", Xtr, []), ("no_port", Xtr_drop, drop_idx)):
            rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=10, max_depth=25,
                                        max_features=0.5, random_state=42 + s,
                                        class_weight="balanced", n_jobs=-1).fit(X, ytr)
            hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                 random_state=13 + s,
                                                 class_weight="balanced").fit(X, ytr)
            result["arms"][f"{arm}_seed{s}"] = {"rf": evaluate(rf, sets, classes, di),
                                                "hgb": evaluate(hgb, sets, classes, di)}
            r = result["arms"][f"{arm}_seed{s}"]["rf"]
            print(f"  {arm} seed{s}: RF korpus FTP {r['corpus_test']['FTP-Patator_class_recall']:.4f} "
                  f"heldout FTP {r['table1_heldout']['FTP-Patator_class_recall']:.4f} "
                  f"esp32ftp atk {r['esp32_ftp_patator_20260814']['attacker_binary']:.4f}", flush=True)
            del rf, hgb

    # tohumlar uzerinden ozet: ortalama +/- std, kol farki
    def agg(arm, model):
        keys = [k for k in result["arms"] if k.startswith(arm + "_")]
        flat = {}
        for k in keys:
            for sname, d in result["arms"][k][model].items():
                for m, v in d.items():
                    if isinstance(v, float):
                        flat.setdefault(f"{sname}.{m}", []).append(v)
        return {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)}
                for m, v in flat.items()}
    result["summary"] = {f"{arm}.{model}": agg(arm, model)
                         for arm in ("with_port", "no_port") for model in ("rf", "hgb")}
    result["provenance"] = common.provenance(("",), [common.DATA_CSV])
    common.write_json("port_retrain.json", result)


if __name__ == "__main__":
    main()
