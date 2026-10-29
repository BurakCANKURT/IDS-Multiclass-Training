# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-10-29
### Added
- First public release of the training and evaluation code, extracted from the
  private development repository.
- Streaming feature engine (41 flow features) and CICIDS-2017 dataset builders.
- Multiclass RF + HGB + LSTM training, ONNX export and OOD-gate calibration.
- Offline and real-world evaluation scripts, pre-registered kill-gate harnesses.
- Welford / live-offline feature parity test.
### Changed
- Data and tool paths are configurable through environment variables
  (`CICIDS_PCAP_ROOT`, `TSHARK`, `IDS_LIVE_DETECTOR`) instead of hard-coded local paths.
