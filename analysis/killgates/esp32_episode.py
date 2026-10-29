# -*- coding: utf-8 -*-
"""
ESP32 held-out: OLAY (epizot) DUZEYI tespit + AKIS-KUMELI guven araliklari.

Hakem itirazi (DG M1 / ND H2-M1, 2026-09-27/28): paket-duzeyi ikili recall
(.2035 / .1172) ve satir-iid Wilson araliklari, (a) bir operatorun gordugu
"saldiri fark edildi mi" sorusuna cevap vermiyor, (b) ayni akisin paketlerini
bagimsiz sayarak araligi yapay olarak daraltiyor. Bu betik ikisini de olcer.

Her set (tek oturum, n_session = 1) icin:
  paket   : ikili recall ve arka plan FAR -- Wilson (eski) VE akis-kumeli bootstrap
  akis    : saldirgan akislarinin >=1 alarm alan orani (akis bootstrap'i)
  zaman   : saldiri penceresi [ilk, son saldirgan paketi]; ilk alarma sure;
            60 s pencerelerin >=1 saldirgan alarmi iceren orani;
            ARKA PLAN icin ayni pencere metrigi (>=1 arka plan alarmi olan 60 s
            pencere orani) ve arka plan alarm/saat
  eslestirilmis: uretim vs .r51_nowin arka plan FAR orani (DG Tablo 5), ayni
            akislar uzerinde eslestirilmis akis bootstrap'i

Kullanim: python analysis/killgates/esp32_episode.py
Cikti   : analysis/killgates/out/esp32_episode.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WINDOW_S = 60.0
ARMS = [("A0_production", ""), ("r51_nowin", ".r51_nowin")]


def load_dump(arm, set_name):
    return pd.read_csv(os.path.join(common.OUT_DIR, "packets", f"{arm}__{set_name}.csv.gz"))


def windows(t, t0, t1):
    """[t0, t1] araligini WINDOW_S'lik pencerelere boler; t'nin pencere indeksleri."""
    n = max(1, int(np.ceil((t1 - t0) / WINDOW_S)))
    return n, np.clip(((t - t0) // WINDOW_S).astype(int), 0, n - 1)


def episode_metrics(df):
    atk, bg = df[df["is_attacker"]], df[~df["is_attacker"]]
    t_cap0, t_cap1 = df["t"].min(), df["t"].max()
    t0, t1 = atk["t"].min(), atk["t"].max()
    atk_alert = atk[atk["alert"]]
    n_w, w_idx = windows(atk_alert["t"].values, t0, t1)
    flow_hit = atk.groupby("flow")["alert"].any()

    # arka plan: tum yakalama boyunca, saldiri penceresi DISI ve ICI ayri
    n_bw, bw_idx = windows(bg.loc[bg["alert"], "t"].values, t_cap0, t_cap1)
    bg_out = bg[(bg["t"] < t0) | (bg["t"] > t1)]
    hours = (t_cap1 - t_cap0) / 3600.0
    # "arka plan" = src != saldirgan; buna KURBANIN saldirgana yanitlari da dahil.
    # Ayristir: saldirgan akislarinin ters yonu vs saldiriyla ILGISIZ trafik.
    atk_flows = set(atk["flow"])
    bg_rev = bg[bg["flow"].isin(atk_flows)]
    bg_unrel = bg[~bg["flow"].isin(atk_flows)]
    # ilgisiz trafik icin pencere metrigi, TUM yakalama boyunca
    n_uw, uw_idx = windows(bg_unrel.loc[bg_unrel["alert"], "t"].values, t_cap0, t_cap1)
    # saldiri penceresi DISINDAKI 60 s pencerelerde ilgisiz alarm (saf FP penceresi)
    out_t = bg_unrel.loc[bg_unrel["alert"] & ((bg_unrel["t"] < t0) | (bg_unrel["t"] > t1)), "t"]
    _, ow_idx = windows(out_t.values, t_cap0, t_cap1)
    _, all_w = windows(np.arange(t_cap0, t_cap1, 1.0), t_cap0, t_cap1)
    _, atk_w = windows(np.arange(t0, t1 + 1, 1.0), t_cap0, t_cap1)
    quiet_windows = sorted(set(all_w) - set(atk_w))
    return {
        "capture_s": float(t_cap1 - t_cap0),
        "attack_window_s": float(t1 - t0),
        "attacker_flows": int(atk["flow"].nunique()),
        "attacker_packets": int(len(atk)),
        "packet_recall_wilson": common.rate(int(atk["alert"].sum()), len(atk)),
        "packet_recall_flow_bootstrap": common.cluster_bootstrap(atk["flow"], atk["alert"]),
        "flow_detection_rate": {
            "k": int(flow_hit.sum()), "n": int(len(flow_hit)),
            "rate": float(flow_hit.mean()),
            "ci95_wilson_flows": list(common.wilson(int(flow_hit.sum()), len(flow_hit)))},
        "time_to_first_alert_s": float(atk_alert["t"].min() - t0) if len(atk_alert) else None,
        "attack_60s_windows": {"n": n_w, "with_alert": int(len(set(w_idx))),
                               "rate": len(set(w_idx)) / n_w},
        "background_packets": int(len(bg)),
        "background_far_wilson": common.rate(int(bg["alert"].sum()), len(bg)),
        "background_far_flow_bootstrap": common.cluster_bootstrap(bg["flow"], bg["alert"]),
        "background_alerts_per_hour": float(bg["alert"].sum() / hours) if hours else None,
        "background_60s_windows_with_alert": {
            "n": n_bw, "with_alert": int(len(set(bw_idx))), "rate": len(set(bw_idx)) / n_bw},
        "background_split": {
            "victim_replies_in_attack_flows": {
                "packets": int(len(bg_rev)), "alerts": int(bg_rev["alert"].sum())},
            "unrelated": {
                "packets": int(len(bg_unrel)), "alerts": int(bg_unrel["alert"].sum()),
                "far_flow_bootstrap": common.cluster_bootstrap(bg_unrel["flow"], bg_unrel["alert"])
                if len(bg_unrel) else None,
                "alerts_per_hour": float(bg_unrel["alert"].sum() / hours) if hours else None,
                "60s_windows_with_alert_whole_capture": {
                    "n": n_uw, "with_alert": int(len(set(uw_idx))),
                    "rate": len(set(uw_idx)) / n_uw},
                "quiet_60s_windows (no attack traffic)": {
                    "n": len(quiet_windows),
                    "with_unrelated_alert": int(len(set(ow_idx) & set(quiet_windows))),
                    "rate": (len(set(ow_idx) & set(quiet_windows)) / len(quiet_windows))
                    if quiet_windows else None}}},
        "background_outside_attack_window": {
            "packets": int(len(bg_out)), "alerts": int(bg_out["alert"].sum()),
            "seconds": float((t0 - t_cap0) + (t_cap1 - t1))},
    }


def main():
    arms = {arm: common.run_realworld_arm(arm, suf, dump=True) for arm, suf in ARMS}
    set_names = [s["name"] for s in arms["A0_production"]["sets"]]
    result = {"note": "n_session = 1 per class; all intervals are within-session "
                      "(flow-clustered), not between-session.",
              "window_s": WINDOW_S, "sets": {}}
    for name in set_names:
        per_arm = {}
        dumps = {arm: load_dump(arm, name) for arm, _ in ARMS}
        for arm, _ in ARMS:
            per_arm[arm] = episode_metrics(dumps[arm])
        # eslestirilmis: ayni paketler, ayni akislar
        a, b = dumps["A0_production"], dumps["r51_nowin"]
        m = a.merge(b[["packet_no", "alert"]], on="packet_no", suffixes=("", "_r51"))
        bg = m[~m["is_attacker"]]
        atk = m[m["is_attacker"]]
        per_arm["paired_r51_vs_production"] = {
            "background_far_ratio": common.cluster_bootstrap(
                bg["flow"], bg["alert"], hits_b=bg["alert_r51"], stat="ratio"),
            "attacker_recall_diff": common.cluster_bootstrap(
                atk["flow"], atk["alert"], hits_b=atk["alert_r51"], stat="diff"),
            "rows_aligned": int(len(m)), "rows_a": int(len(a)), "rows_b": int(len(b)),
        }
        result["sets"][name] = per_arm
        p = per_arm["A0_production"]
        print(f"{name}: paket recall {p['packet_recall_wilson']['rate']:.4f} "
              f"wilson {np.round(p['packet_recall_wilson']['ci95'], 4)} "
              f"akis-boot {np.round(p['packet_recall_flow_bootstrap']['ci95'], 4)} | "
              f"akis tespit {p['flow_detection_rate']['k']}/{p['flow_detection_rate']['n']} | "
              f"ilk alarm {p['time_to_first_alert_s']} s | "
              f"60s pencere {p['attack_60s_windows']['with_alert']}/{p['attack_60s_windows']['n']} "
              f"| bg alarm/saat {p['background_alerts_per_hour']:.1f}", flush=True)
    result["provenance"] = common.provenance(("", ".r51_nowin"))
    common.write_json("esp32_episode.json", result)


if __name__ == "__main__":
    main()
