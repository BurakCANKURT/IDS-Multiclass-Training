# Multiclass Network IDS — Training and Generalization Study

Training, evaluation and generalization experiments for a multiclass network intrusion
detector trained on [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html).
This repository produces the models used by the live detection console
(**`<live-console repo link>`**).

This project started as my bachelor's thesis. The goal was to train models on CICIDS-2017
and then maximize how well they detect attacks in traffic they have never seen.

Output classes: `BENIGN, DDoS, DoS, FTP-Patator, PortScan, SSH-Patator, WebAttack`,
using 41 streaming flow features.

## Pipeline

```
CICIDS-2017 pcaps
  -> analysis/pcap_validation_pilot*.py           streaming flow-feature engine (41 features)
  -> analysis/build_streaming_dataset.py           batch composition
  -> multiclass_training/build_multiclass_streaming_dataset.py
        -> multiclass_training/data/multiclass_streaming_dataset.csv
  -> multiclass_training/train_multiclass_models.py   Random Forest + LSTM
     multiclass_training/train_multiclass_hgb.py      HistGradientBoosting
  -> analysis/convert_multiclass_committee_to_onnx.py ONNX export (used at runtime)
     analysis/mc_ood_calibrate.py                     reference set for the OOD gate
  -> multiclass_training/models/
```

**Why a streaming dataset instead of the CICIDS CSVs?** The official CSVs come from
CICFlowMeter, which computes features once per finished flow. A live detector sees
packets one at a time and has to compute features incrementally. Training on CSV features
and running on streaming features would mean training and testing on two different
distributions. The models are therefore trained on features produced by the same
streaming engine that runs live.

## Results

### Held-out detection (binary-collapsed protocol)

Held-out batches come from the **same CICIDS capture session and are separated in time**
from the training batches; this is not cross-day or cross-session data (see Limitations).
MCC intervals are 95% bootstraps over whole batches (2,000 resamples).

| Family | DR | FAR | Precision | MCC [95% CI] |
|---|---|---|---|---|
| FTP-Patator | 0.9725 | 0.0023 | 0.9845 | 0.9752 [0.945, 0.987] |
| SSH-Patator | 0.9795 | 0.0026 | 0.9896 | 0.9807 [0.975, 0.985] |
| WebAttack | 0.9687 | 0.0030 | 0.9660 | 0.9645 [0.961, 0.970] |
| PortScan | 0.9966 | 0.0005 | 0.9998 | 0.9938 [0.837, 0.999] |
| DDoS | 0.9697 | 0.0308 | 0.9949 | 0.8841 [0.604, 0.912] |
| **Overall (in-scope / all groups)** | 0.9801 / 0.8961 | 0.0038 / 0.0038 | 0.9877 / 0.9857 | 0.9789 / 0.9236 |

Leakage check: the run was repeated on batches at least 10 indices (about 1–15 min) away
from every training batch. DR and FAR stayed at the same level, with a largest drop of
2 points (FTP DR .973 → .954).

### Generalization study: interventions tested and rejected

The models are trained on 2017 traffic but meant to run on today's traffic. Each candidate
fix for that gap was tested against a **pre-registered kill gate** before any production
change. Most were rejected:

| Idea | Outcome |
|---|---|
| Active flow timeout (match 2017 flow lifetimes) | rejected: the apparent 2017 lifetime limit (~2.5 s) was a 10k-packet batching artefact |
| IPv6 feature parity | closed: real capture was 0.28% IPv6, all link-local ICMPv6 |
| Drop `Init_Win_bytes_*` (TCP window shortcut), RF+HGB retrain | rejected: background FAR rose 2.41× [2.10, 2.83] |
| Conditional CORAL domain-invariant LSTM | rejected: feature supports were disjoint (AUC 1.0) |
| Support-preserving nuisance randomisation | rejected at a zero-training gate: correct-class rate .0000 |
| Drop port bucket, retrain (3 seeds) | binary detection kept (≥ 99.7%), but FPR rose 3.6×, not adopted |

The harnesses that reproduce these decisions are in [`analysis/killgates/`](analysis/killgates/).
Every output JSON records the SHA-256 of its inputs and the git commit it ran on.

## Reproducing

Requirements: Python 3.11, [Wireshark](https://www.wireshark.org/) (`tshark`), and the
CICIDS-2017 pcaps (not included).

```
pip install -r requirements.txt
```

Paths are set with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CICIDS_PCAP_ROOT` | `data/batch_pcaps` | folder with the per-day CICIDS-2017 pcap batches |
| `TSHARK` | `C:\Program Files\Wireshark\tshark.exe` | tshark executable |
| `IDS_LIVE_DETECTOR` | — | path to the live detector, only needed for the parity test |
| `IDS_PARITY_PCAP` | `data/realworld/...` | any real capture with more than 1,000 IP packets, for the parity test |

Typical order:

```
python multiclass_training/build_multiclass_streaming_dataset.py
python multiclass_training/train_multiclass_models.py
python multiclass_training/train_multiclass_hgb.py
python analysis/convert_multiclass_committee_to_onnx.py
python analysis/mc_ood_calibrate.py
python multiclass_training/offline_test_multiclass.py
python analysis/killgates/run_all.py
```

Pretrained models are attached to the GitHub Release instead of being stored in git.

## Tests

```
python -m pytest tests -q
```

`tests/test_welford_flow_parity.py` checks that the live detector's incremental (Welford)
feature computation produces the same 41 features as the offline reference engine.
The live detector keeps its own copy of the feature code, so this test is what guards
against training/serving skew. It needs the live-console repository
(`IDS_LIVE_DETECTOR`) and a real capture (`IDS_PARITY_PCAP`), and is skipped without them.

## Limitations

- **One capture session per attack in CICIDS-2017.** A real cross-session test is impossible
  on this data; all held-out numbers are within a single session.
- **Duplicated logic across repositories.** The feature engine and the suppression gates exist
  both here (offline evaluation) and in the live detector. A change on one side has to be
  mirrored on the other; the parity test only covers the features, not the gates.
- `pcap_validation_pilot.py` through `_v5.py` are successive versions that import each other
  and are all still part of the dataset-building chain; consolidating them is planned.
- **Real-world captures are not included.** The home-lab captures behind the real-world and
  kill-gate results contained private network traffic and are not published. Those numbers can only be reproduced with your own
  captures; lab IP addresses in the scripts refer to the original home lab network.
- Code comments and the detailed notes in `docs/` are in Turkish.

## Repository layout

| Path | Contents |
|---|---|
| `analysis/` | feature engine, dataset builders, ONNX export, OOD calibration, validation |
| `analysis/killgates/` | pre-registered kill-gate harnesses |
| `multiclass_training/` | dataset builders, training, offline evaluation |
| `tests/` | feature parity test |
| `docs/` | generalization audit, literature comparison, v5 strategy (Turkish) |

## How this was built

I designed the architecture, defined the experiments and kill gates, and made the design
decisions. Most of the code was written with AI coding agents under my direction; I reviewed
and validated the results.

## License

MIT, see [LICENSE](LICENSE). CICIDS-2017 is provided by the Canadian Institute for
Cybersecurity under its own terms.
