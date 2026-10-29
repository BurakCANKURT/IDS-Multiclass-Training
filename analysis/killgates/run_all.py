# -*- coding: utf-8 -*-
"""
Tum kill-gate harness'lerini sirayla kosar ve out/SUMMARY.json yazar.

    python analysis/killgates/run_all.py            # hepsi (~1.5-2 saat, tek cekirdek-agirlikli)
    python analysis/killgates/run_all.py r51 spnr   # secili olanlar

Gercek-dunya kollari out/_rw_*.json'a onbelleklenir (ayni suffix+override ise
tekrar kosulmaz); temiz kosu icin once `out/` dizinini silin.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GATES = {  # kisa ad -> (betik, cikti json, makale referansi)
    "r38":   ("r38_phase_recall.py", "r38_phase_recall.json", "DG Table 4"),
    "r51":   ("r51_nowin.py", "r51_nowin.json", "DG Table 5"),
    "gate35": ("gate35_initwin_bwd.py", "gate35_initwin_bwd.json", "DG Sec. 4.5 / 5"),
    "spnr":  ("spnr_gate_n1.py", "spnr_gate_n1.json", "DG Sec. 4.5"),
    "o3":    ("gate_o3_flowshape.py", "gate_o3_flowshape.json", "DG Sec. 4.9 / 5"),
    "r50":   ("r50_coral_gate0.py", "r50_coral_gate0.json", "DG Table 6, Table 7"),
    "at0":   ("gate_at0_active_timeout.py", "gate_at0_active_timeout.json", "DG Table 1"),
}


def main():
    wanted = sys.argv[1:] or list(GATES)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="3")
    summary = {}
    for g in wanted:
        script, out, ref = GATES[g]
        print(f"\n===== {g} ({ref}) =====", flush=True)
        rc = subprocess.run([sys.executable, os.path.join(common.THIS_DIR, script)],
                            env=env, cwd=common.PROJECT_DIR).returncode
        entry = {"script": script, "paper": ref, "returncode": rc}
        if rc == 0:
            r = common.read_json(out)
            entry["verdict"] = r.get("verdict")
            entry["fidelity_all_ok"] = r.get("fidelity_all_ok")
        summary[g] = entry
    common.write_json("SUMMARY.json", summary)
    for g, e in summary.items():
        print(f"{g:7s} rc={e['returncode']} verdict={e.get('verdict')} "
              f"fidelity={e.get('fidelity_all_ok')}")


if __name__ == "__main__":
    main()
