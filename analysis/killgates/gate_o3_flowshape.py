# -*- coding: utf-8 -*-
"""
GATE-O3 -- akis-SEKLI alt-uzayinda ESP32'nin en yakin sinifi (SIFIR EGITIM,
model YUKLENMEZ -- bu bir geometri olcumudur).

On-kayitli catal:
  ESP32 kisa-akis WebAttack'i sekil-uzayinda WebAttack'a yakinsa -> confound,
  retrain'in sansi var; degilse -> yapisal blokaj, DUR (Init_Win/port-dusurulmus
  retrain KOSULMAZ).

Olcumler
  1. Dengeli 1-NN (sinif basina 4000 referans) -- 3 alt-uzay x 2 donusum.
  2. Dogal-onselli k=50 bolge sahipligi (tam SYN-rejim korpusu), SHAPE32|z.
     Akis-kumeli sifir icin 3/n ust siniri.
  3. Curutme-1: selfgen kisa-akis WebAttack'i (cihaz imzasi yok) ayni olcumde.
  4. Curutme-2: korpus WebAttack'i, kendi katmani (kisa/uzun) referanstan
     cikarilarak (leave-stratum-out); ve batch-tutulan kontrolun ezber mesafesi.
  5. Mekanizma: medyan akis-sekli ozellikleri.

Orijinal (2026-08-21) harness kayboldu; bu betik verdict memosundan yeniden
kuruldu. Alt-ornekleme tohumlari farkli olabilecegi icin rakamlar memo ile
BIREBIR olmak zorunda degildir -- karar kurali niteldir (WebAttack payi taban
onselin altinda mi) ve `memo_reference` alaninda yan yana raporlanir.

Kullanim: python analysis/killgates/gate_o3_flowshape.py
Cikti   : analysis/killgates/out/gate_o3_flowshape.json
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLS = common.raw_columns()
PORT = [c for c in COLS if c.startswith("PortBucket_")]
INIT_WIN = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
NARROW = ["PSH Flag Count", "ACK Flag Count", "SYN Flag Count", "FIN Flag Count",
          "RST Flag Count", "act_data_pkt_fwd", "Subflow Bwd Packets",
          "Flow IAT Mean", "Fwd IAT Max", "Bwd IAT Total", "Bwd IAT Max", "Flow IAT Max",
          "Down/Up Ratio", "Fwd/Bwd Byte Ratio", "Flow Bytes/s", "Flow Packets/s",
          "Bwd Packets/s"]
SUBSPACES = {"ALL41": COLS,
             "SHAPE32": [c for c in COLS if c not in PORT + INIT_WIN],
             "NARROW": NARROW}
assert len(SUBSPACES["SHAPE32"]) == 32 and len(NARROW) == 17
N_PER_CLASS = 4000
K = 50
MEMO = {
    "esp32_balanced_1nn_webattack": {"ALL41|z": 0.0, "ALL41|q": 0.0, "SHAPE32|z": 0.0,
                                     "SHAPE32|q": 0.0, "NARROW|z": 0.0004, "NARROW|q": 0.0013},
    "esp32_region_share_k50": {"BENIGN": 0.7981, "PortScan": 0.1524, "DDoS": 0.0465,
                               "DoS": 0.0030, "WebAttack": 0.0},
    "selfgen_region_webattack": 0.0104,
    "leave_stratum_out_webattack": {"short": 0.0068, "long": 0.0039},
    "heldout_nn_dist_p50": 0.0023, "heldout_nn_dist_lt_0.01": 0.946,
    "median_act_data_pkt_fwd": {"corpus_WebAttack": 23, "esp32": 1, "selfgen": 1,
                                "corpus_DoS": 1, "corpus_BENIGN": 3},
}


def esp32_attacker_features():
    """offline_test_multiclass.run_offline ANA PASI ile ayni ozellik cikarimi."""
    import contextlib
    import io
    import pcap_validation_pilot as v1
    import validate_realworld_pcaps as vr
    with contextlib.redirect_stdout(io.StringIO()):
        import offline_test_multiclass as mc
    spec = [s for s in vr.REALWORLD_SETS if s["name"] == "esp32_webattack_20260814"][0]
    packets = v1.read_pcap(spec["pcap"])
    ft = mc.UniversalRealTimeFlowTableV5()
    rows, flows = [], []
    for p in packets:
        if mc._is_non_ip(p):
            continue
        mf, fk = ft.extract_universal_features(p)
        if mf is None or p.get("ip.src") != spec["attacker_ip"]:
            continue
        rows.append([float(mf.get(c, 0.0)) for c in COLS])
        flows.append(str(fk))
    return pd.DataFrame(rows, columns=COLS), np.array(flows)


def make_transform(kind, ref):
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    t = StandardScaler() if kind == "z" else QuantileTransformer(
        n_quantiles=1000, output_distribution="uniform", subsample=200_000,
        random_state=common.SEED)
    t.fit(ref)
    return lambda X: np.nan_to_num(t.transform(X))


def share(labels):
    c = Counter(labels)
    n = sum(c.values())
    return {k: round(v / n, 4) for k, v in sorted(c.items(), key=lambda x: -x[1])}


def main():
    from sklearn.neighbors import NearestNeighbors

    df = pd.read_csv(common.DATA_CSV)
    ref = df[df["SYN Flag Count"] > 0].reset_index(drop=True)
    y = ref["label_str"].values
    prior = share(y)
    rng = np.random.RandomState(common.SEED)

    esp, esp_flows = esp32_attacker_features()
    sg = pd.read_csv(os.path.join(common.MC_DIR, "data", "selfgen_dataset.csv"))
    sg_wa = sg[sg["label_str"] == "WebAttack"]
    print(f"ref(SYN)={len(ref)} esp32={len(esp)} ({len(set(esp_flows))} akis) "
          f"selfgenWA={len(sg_wa)}", flush=True)

    # --- 1. dengeli 1-NN, 6 kombinasyon ---
    bal_idx = np.concatenate([
        rng.choice(np.where(y == c)[0], min(N_PER_CLASS, (y == c).sum()), replace=False)
        for c in sorted(set(y))])
    balanced, region_all = {}, {}
    for sname, cols in SUBSPACES.items():
        for tk in ("z", "q"):
            tf = make_transform(tk, ref[cols].values)
            Xr = tf(ref[cols].values)
            nn = NearestNeighbors(n_neighbors=1).fit(Xr[bal_idx])
            out = {}
            for qname, q in (("esp32", esp), ("selfgen_WebAttack", sg_wa)):
                _, ix = nn.kneighbors(tf(q[cols].values))
                out[qname] = share(y[bal_idx][ix[:, 0]])
            balanced[f"{sname}|{tk}"] = out
            # dogal-onsel k=50 bolge payi, esp32 (her kombinasyon)
            _, ix = NearestNeighbors(n_neighbors=K).fit(Xr).kneighbors(tf(esp[cols].values))
            region_all[f"{sname}|{tk}"] = share(y[ix].ravel())
            print(f"  {sname}|{tk}: esp32 WA={out['esp32'].get('WebAttack', 0.0)}", flush=True)

    # --- 2/3. dogal-onsel k=50 bolge sahipligi, SHAPE32|z ---
    cols = SUBSPACES["SHAPE32"]
    tf = make_transform("z", ref[cols].values)
    Xref = tf(ref[cols].values)
    nn50 = NearestNeighbors(n_neighbors=K).fit(Xref)

    def region(q):
        _, ix = nn50.kneighbors(tf(q[cols].values))
        return y[ix], ix

    lab, _ = region(esp)
    esp_region = share(lab.ravel())
    first_nn = lab[:, 0]
    flow_hit = pd.Series(first_nn == "WebAttack").groupby(esp_flows).any()
    n_flows = int(len(flow_hit))
    sg_lab, _ = region(sg_wa)
    sg_short = sg_wa[sg_wa["act_data_pkt_fwd"] <= 2]
    sg_short_lab, _ = region(sg_short)

    # --- 4. leave-stratum-out: korpus WebAttack kisa / uzun ---
    lso = {}
    wa = ref["label_str"] == "WebAttack"
    for stratum, m in (("short", wa & (ref["act_data_pkt_fwd"] <= 2)),
                       ("long", wa & (ref["act_data_pkt_fwd"] > 2))):
        keep = ~m.values
        nn_s = NearestNeighbors(n_neighbors=K).fit(Xref[keep])
        _, ix = nn_s.kneighbors(Xref[m.values])
        lso[stratum] = {"n": int(m.sum()), "share": share(y[keep][ix].ravel())}

    # ezber kontrolu: batch-tutulan WebAttack -> egitim WebAttack'ina en yakin mesafe
    tr, te, _ = common.load_corpus_split()
    tr_wa = tr[(tr["label_str"] == "WebAttack") & (tr["SYN Flag Count"] > 0)]
    te_wa = te[(te["label_str"] == "WebAttack") & (te["SYN Flag Count"] > 0)]
    d, _ = NearestNeighbors(n_neighbors=1).fit(tf(tr_wa[cols].values)).kneighbors(
        tf(te_wa[cols].values))
    d = d[:, 0]

    # --- 5. mekanizma ---
    mech_cols = ["act_data_pkt_fwd", "PSH Flag Count", "ACK Flag Count",
                 "Total Length of Fwd Packets", "Flow IAT Mean"]
    groups = {"corpus_WebAttack": ref[wa], "corpus_BENIGN": ref[y == "BENIGN"],
              "corpus_DoS": ref[y == "DoS"], "corpus_PortScan": ref[y == "PortScan"],
              "selfgen_WebAttack": sg_wa, "esp32_WebAttack": esp}
    mech = {g: {"n": int(len(v)), **{c: float(v[c].median()) for c in mech_cols}}
            for g, v in groups.items()}

    wa_prior = prior.get("WebAttack", 0.0)
    esp_wa = esp_region.get("WebAttack", 0.0)
    ub = 3.0 / n_flows if int(flow_hit.sum()) == 0 else None
    near_webattack = max(esp_wa, max(v.get("WebAttack", 0.0)
                                     for v in region_all.values())) >= wa_prior
    result = {
        "gate": "GATE-O3 flow-shape subspace nearest class (zero-training)",
        "preregistered_fork": "near WebAttack -> confound (retrain may help); "
                              "else -> structural block, STOP",
        "reference": {"rows_syn_regime": int(len(ref)), "class_prior": prior},
        "queries": {"esp32_attacker_rows": int(len(esp)), "esp32_flows": n_flows,
                    "selfgen_webattack_rows": int(len(sg_wa))},
        "balanced_1nn": balanced,
        "esp32_region_share_k50_all_combos": region_all,
        "esp32_region_webattack_max_over_combos": max(
            v.get("WebAttack", 0.0) for v in region_all.values()),
        "region_share_k50_SHAPE32_z": {
            "esp32": esp_region,
            "esp32_enrichment_vs_prior": {c: round(v / prior[c], 2) for c, v in esp_region.items()},
            "esp32_flows_with_webattack_1nn": int(flow_hit.sum()),
            "esp32_flow_level_upper_bound_3_over_n": ub,
            "selfgen_WebAttack": share(sg_lab.ravel()),
            "selfgen_WebAttack_short": share(sg_short_lab.ravel()),
        },
        "leave_stratum_out": lso,
        "heldout_memorization": {"n": int(len(d)), "p50": float(np.median(d)),
                                 "p90": float(np.percentile(d, 90)),
                                 "frac_lt_0.01": float((d < 0.01).mean()),
                                 "frac_lt_0.05": float((d < 0.05).mean())},
        "mechanism_medians": mech,
        "verdict": "PROCEED (near WebAttack)" if near_webattack
                   else "STRUCTURAL BLOCK -- STOP",
        "memo_reference": MEMO,
        "provenance": common.provenance((), [common.DATA_CSV,
                                             os.path.join(common.MC_DIR, "data", "selfgen_dataset.csv")]),
    }
    common.write_json("gate_o3_flowshape.json", result)
    print(f"esp32 bolge={esp_region} (WA taban {wa_prior}) -> {result['verdict']}")


if __name__ == "__main__":
    main()
