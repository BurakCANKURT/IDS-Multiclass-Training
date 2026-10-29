# Kill-gate harness'leri

DG (`dataset_generalization_article`) ve ND makalelerindeki on-kayitli
kill-gate'lerin **yeniden uretilebilir** betikleri. Orijinalleri 2026-08-20 ..
2026-09-01 arasinda oturum scratchpad'lerinde kosulmus ve kaybolmustu; bu dizin
verdict memolarindan (`.claude/agent-memory/*/project_*verdict*.md`) yeniden
kuruldu ve her betik memodaki rakamlarla **sadakat kontrolu** yapar.

Hicbir betik uretim model dosyasina, `results/`'a veya egitim CSV'sine yazmaz.
Ciktilar `out/` altina gider (her JSON'da girdi dosyalarinin SHA-256'si ve
git HEAD'i vardir).

```
python analysis/killgates/run_all.py          # hepsi
python analysis/killgates/run_all.py r51 o3   # secili
python -m pytest tests/test_welford_flow_parity.py -q
```

| kisa ad | betik | makale | soru | sonuc | memo ile |
|---|---|---|---|---|---|
| r38 | `r38_phase_recall.py` | DG Tablo 4 | WebAttack recall'u payload fazina gore | monoton cokus (RF .946 -> .306 -> .172) | **birebir** (n ve 4 hane) |
| r51 | `r51_nowin.py` | DG Tablo 5 | Init_Win dusurulmus RF+HGB retrain | GATE-A gecti, B ve C kaldi -> RED | **birebir**; `--retrain` sifirdan egitimde argmax %100 ayni |
| gate35 | `gate35_initwin_bwd.py` | DG 4.5 | Init_Win_bwd kurban parmak izi mi | 6 saldiri-kapasiteli / 41 saf-benign deger; esp32_ftp .1172 -> .6007; RF korpus .9324 -> .4113 | **birebir** (7 kol + 4 korpus kolu; C1 permutasyonu tohum farki .0029) |
| spnr | `spnr_gate_n1.py` | DG 4.5 | Init_Win'i korpus WebAttack degerine sabitlemek dogru-sinif uretir mi | dogru-sinif .0000 dort kolda -> KILL; T5f == T1 | **birebir** |
| o3 | `gate_o3_flowshape.py` | DG 4.9 / 5 | ESP32 kisa-akis WebAttack'i sekil-uzayinda hangi sinifa yakin | BENIGN; WebAttack payi taban onselin altinda -> DUR | **karar ayni**, ara rakamlar yaklasik (asagiya bkz.) |
| r50 | `r50_coral_gate0.py` | DG Tablo 6, 7 | kosullu CORAL / CDAN domain-AUC'u dusuruyor mu | tum kollar AUC ~1.000 -> Gate-0 KALDI | Tablo 7 ve kok-neden tablosu **birebir** (3 hane); Tablo 6 ~1.000 |
| at0 | `gate_at0_active_timeout.py` | DG Tablo 1 | aktif akis zaman asimi | K1 gecti (trivial), K2 KALDI, K3 gecti -> RED | K1 birebir, K2 yakin; **yan-etki DUZELTMESI** asagida |

## Ortak altyapi

* `common.py` -- korpus bolmesi (`train_multiclass_models.batch_level_split`
  import edilir, kopyalanmaz), Wilson CI, model yukleme (`MC_EVAL_SUFFIX`
  sonekleri), SHA-256 provenans.
* `realworld.py` -- tek bir gercek-dunya kolu (alt-surec). Metrik
  `analysis/validate_realworld_pcaps.py` ile birebir (registry, GT kurali
  `src_ip == attacker_ip`, `predicted_class`). `--override OZELLIK=DEGER`
  yalniz saldirgan satirlarina ve SYN-gorulmus akislara uygulanir; arka plan
  FAR'i her kolda sabit kalmalidir (kapsam kaniti). Kollar `out/_rw_*.json`'a
  onbelleklenir.
* Taban sadakati: A0 kolu registry baseline'ini birebir uretir
  (.2035 / .1172; davranissal 376/32, hacimsel 108/0; arka plan FAR
  .00635 / .00252). Korpus: RF .9324 / FPR .00417, HGB .9680, LSTM .9853.

## Yeniden kurulumdan kalan farklar (durustluk kaydi)

* **o3:** karar kurali ve ana olcumler memo ile ayni (ESP32 WebAttack
  bolge payi .0000, 0/394 akis, 3/n ust siniri .0076 < taban .0140; PortScan
  alti kombinasyonda .0000). Ara rakamlar farkli: memo'nun dogal-onsel bolge
  payi (BENIGN .80 / PortScan .15) hangi alt-uzayda hesaplandigi belgelenmemis;
  betik alti kombinasyonun hepsini raporlar. Leave-stratum-out ve ezber
  mesafesi (memo p50 .0023 / %94.6 < .01) de ayni yonde ama farkli buyuklukte.
* **at0 -- MAKALE DUZELTMESI GEREKIR:** memodaki ve DG Tablo 1'deki
  "canli yanlis alarm 17x artti (%0.086 -> %1.456)" rakami tam uretim hattinin
  degil, K3'teki davranissal-dal PROXY'sinin (LSTM argmax + veto + OOD<=30)
  alarm oranidir. Olculdu: T=inf'de proxy 59 alarm (%0.084, memo 60 ile uyumlu)
  vs tam `run_offline` hatti 12 alarm; T=3'te proxy 644 (~11x) vs tam hat 15
  (1.25x). Immature/early_attack kapilari bu yan etkinin neredeyse tamamini
  emiyor. Verdict (K2 kaldi) degismiyor, ama "17x" ifadesi "davranissal dal
  11-17x; kapili uretim hatti ~1.25x" olarak duzeltilmeli. K3 mutlak sayilari
  (TP 956 vs memo 2925) proxy maskesinin belgelenmemis ayrintisi nedeniyle
  farkli; yon ayni (DR artiyor).
* **r50 / at0:** orijinal protokolun belgelenmemis ayrintilari (alt-ornek
  boyu, aralik-ortusmesi tanimi, K3 proxy maskesi) makul varsayimlarla
  dolduruldu; her JSON `memo_reference` alaninda orijinal rakamlari tasir.
* **esp32 (Tablo 4 son satir):** makaledeki n=16.322, artik diskte olmayan eski
  bir halka-tamponu yakalamasindandi; betik kalici registry setini (n=2.378)
  kullanir.

## Yeniden kurulMAyanlar

* **DG Tablo 2 (PG-6, TLS-583 capraz-testbed lift):** santos tarafinin
  es-yerlesik benign istemci kumesi (NW) ve kullanilan siniflandirici memoda
  belgelenmemis; 847 MB wardbeck pcap'i mevcut `read_pcap` ile bellege tam
  yuklenir. Ham veri `D:\tez_data\ait_{santos,wardbeck}_extract\` altinda
  duruyor. TODO madde 7 (uretim modelinde TLS-583 karsi-olgusu) bu acigin
  uretim-ilgili halini dogrudan olcer.

## Son kosu durumu (2026-09-28)

`out/` icindeki r38, r51, gate35, spnr, r50 JSON'lari guncel betiklerle tam
kosudur. `gate_at0_active_timeout.json` yalnizca `--quick` (T in {inf, 3}) ve
proxy metriginden ONCEKI surumle uretildi; `gate_o3_flowshape.json` alti
kombinasyonluk bolge payi eklenmeden ONCEKI surumle. Ikisinin tam kosusu sistem
bellegi azaldigi icin durduruldu -> `python analysis/killgates/run_all.py at0 o3`
ile tamamlanmali (baska agir surec calismiyorken).
