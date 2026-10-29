# v5 Stratejisi — Performans VE Genellenebilirligi BIRLIKTE Artirma

(GENERALIZATION_AUDIT.md'nin devami. Hedef: Finding 2 [Destination Port shortcut]
duzeltilirken, sadece "DR'lerin dustugunu kabullenmek" degil, kaybedilen sinyali
gercekten davranissal/genellenebilir ozelliklerle telafi etmek.)

## Dogrulanmis kod gercekleri (tasarimin dayandigi zemin)

Tum konumlar `pcap_validation_pilot.py` uzerinde dogrulandi:

| Gercek | Konum | v5 icin anlami |
|---|---|---|
| `Destination Port` = `float(flow["dst_port"])`, ham | satir 363 | #1 importance shortcut; bucket'lanacak |
| `Active Max`/`Active Min` = sabit `0.0` | satir 390 | 2/32 boyut OLU — bedavaya degistirilebilir |
| Sadece PSH/ACK sayiliyor | satir 318-321 | SYN/FIN/RST YOK — davranissal sinyal eksik |
| `flags_int = _parse_tcp_flags(...)` zaten var | satir 317 | SYN/FIN/RST eklemek 3 satir `if flags_int & 0xNN` |
| `update_volumetric` zaten SYN (`0x02`) parse ediyor | satir 188-197 | SYN sayim mantigi kanitlanmis |
| Flow rate'leri MUTLAK (`/flow_duration_sec`) | satir 357-360 | bandwidth-bagimli; oran-tabanli ek onerilir |
| Preprocessing = TEK `StandardScaler` | retrain_v4 58-60 | one-hot 0/1 ek ColumnTransformer GEREKTIRMEZ |
| Feature sirasi `bundle["raw_columns"]`'a kilitli | satir 477 | yeni ozellik = yeni bundle+dataset+egitim |
| Dispatcher `cfg["RAW_COLUMNS"]`'i DINAMIK okur | satir 521/531/553 | yeni kolonlar OTOMATIK akar |
| LSTM girdisi uyumsuzlukta SESSIZCE pad/truncate | **satir 559-564** | **FOOTGUN**: yanlis kolon crash etmez, yanlis deger besler |
| `dst_port` dispatcher ciktisinda kullaniliyor | satir 586 | port'u feature'dan cikarsan bile flow state'te tut |

**Kritik footgun (559-564):** LSTM, egitildiginden farkli sayida ozellik gelirse
hata vermez; sifir-pad'ler veya keser. Ozellik sayisi degisirse `MainLSTM_v5`
MUTLAKA yeni setle yeniden egitilmeli, `n_feat_lstm` dogru olmali. Bu, v5
retrain'inin atlanamaz olmasinin teknik sebebi.

---

## 1. Port feature yeniden tasarimi (EN YUKSEK ONCELIK)

**1a. Kategorik bucketing — YAP.** Ham portu kaldir, 7-kovali one-hot koy:

```python
def _port_bucket(p):
    p = int(p)
    if p == 21:          return "ftp"
    if p == 22:          return "ssh"
    if p in (80, 8080):  return "http"   # WebAttack, DDoS, Bot-C2(8080) burada erir
    if p == 443:         return "https"
    if p == 53:          return "dns"
    if p < 1024:         return "other_wk"   # 444 buraya duser
    return "ephemeral"                       # 54119/54131 buraya duser
```

**1b. Port-randomizasyon augmentasyonu — KACIN.** One-hot bucketing ZATEN
kova-ici tum portlari ayni degere indirgediginden randomizasyon **gereksizdir
ve yeni shortcut riski yaratir** (model "kova-sinir gurultu desenini"
ogrenebilir). Ham port + randomizasyon ayrica StandardScaler altinda
54119-vs-80 olcek sorunu yaratir. **Secenek A: saf one-hot, augmentasyon yok**
— en durust.

**Skeptik not:** Bucketing de yumusak bir shortcut (Finding 1 tek-oturum
duzelmiyor), ama "ephemeral kova" shortcut'i "literal 54119" ezberinden
onemli olcude daha genellenebilir.

**Kod:** satir 363'u kaldir, master_features'a `PortBucket_*` 7 anahtari
ekle. `flow["dst_port"]` flow state'te KALMALI (satir 586 ciktisi ve satir 223
`update_progressive` TLS_PORTS kontrolu kullaniyor) — sadece `raw_columns`'tan
cikar.

### EMPIRIK DOGRULAMA (bu oturumda yapildi, `streaming_training_dataset_v4.csv` uzerinde)

Bucketing sonrasi label=1 vs label=0 dagilim (en yuksek 1-2 kova):

| family | label=1 dominant bucket(s) | label=0'da ayni kova | yorum |
|---|---|---|---|
| FTP-Patator | ftp=93.2% | ~0% (top4'te yok) | **MUKEMMEL korunuyor** |
| SSH-Patator | ssh=95.2% | ~0% (top4'te yok) | **MUKEMMEL korunuyor** |
| WebAttack | http=100% | http=20.1% | onceki gibi kismi (degisim yok) |
| **Infiltration** | **ephemeral=82.6%** | **ephemeral=5.3%** | **Beklenenden DAHA AZ yikici** — 82.6 vs 5.3, hala anlamli ayristirma kaliyor |
| Bot | http=100% | **http=35.9%** | **EN ZAYIF** — %36 false-positive-rate riski, Bolum 2'ye bagimli |
| **PortScan** | **ephemeral=84.3%** | **~0% (top4'te yok)** | **BONUS — ham portla yoktu, bucket'lamayla ORTAYA CIKAN yeni, genellenebilir sinyal!** |
| DDoS | http=96.6% | http=21.3% (n=94 kucuk) | makul korunuyor |
| DoS (Wed) | http=88.5% | http=14.7% | korunuyor (mevcut MCC=0.952 ile tutarli) |

**Sonuc**: Infiltration icin "82.6% vs 5.3%" hala guclu bir sinyal — agent'in ilk
tahmini (DR ~0.3-0.6) muhtemelen ASIRI PESIMIST; bucketing + Bolum 2
ozellikleriyle DR 0.6-0.85 araliginda kalabilir (garanti degil, retrain
gerekir). PortScan icin bucketing SADECE KAZANC (yeni ayristirici sinyal,
maliyetsiz). Bot, port-shortcut kaybindan en cok zarar gorecek aile —
Bolum 2'nin SYN/RST/Down-Up'i Bot icin KRITIK.

---

## 2. Olu/eksik ozellik — EN IYI MALIYET/DEGER, ONERILEN ILK ADIM

`Active Max`/`Active Min` (sabit 0.0) yerine GERCEK davranissal sinyal.
satir 318-321'i genislet:

```python
if flags_int & 0x02: flow["syn_count"] += 1   # SYN
if flags_int & 0x01: flow["fin_count"] += 1   # FIN
if flags_int & 0x04: flow["rst_count"] += 1   # RST
```

master_features'a (satir 390 yerine): `"SYN Flag Count"`, `"FIN Flag Count"`,
`"RST Flag Count"`, ve oran-tabanli `"Down/Up Ratio": bwd_packets / max(fwd_packets, 1)`.

| aile | en cok yardim eden (ON TAHMIN) | gerekce |
|---|---|---|
| FTP/SSH-Patator | **RST Count** | basarisiz auth -> sunucu RST; davranissal + genellenebilir |
| PortScan | **SYN Count + Down/Up** | SYN-scan: cok SYN/az yanit -> dusuk Down/Up |
| DDoS/SYN-flood | **SYN Count** | model-seviyesi sinyali is_flood ile tutarli hale getirir (FAR'a iyi) |
| Bot | Down/Up (zayif) | port shortcut'i gidince tek umut; MCC 0.43'u biraz kurtarir |
| Infiltration | belirsiz | net flag imzasi yok — durust risk, bucket kaybini tam telafi etmeyebilir |

**Neden ilk adim:** Bolum 1 ile AYNI retrain pasinda gider (ek maliyet ~sifir),
port-bucketing kaybini telafi edecek motor budur, gercekten davranissal
(kimlik tasimaz) oldugundan "reported vs true DR" acigini kapatir. TEK BASINA
bile (port'a dokunmadan) risksiz kazanc verir.

### EMPIRIK DOGRULAMA #2 (bu oturumda yapildi, pcap'lerden YENIDEN CIKARIM)

`_check_v5_new_features.py`: v4'un EGITIM kompozisyonuyla (AYNI batch'ler,
`build_streaming_dataset_v3.py` TOP_K=3 + `build_infiltration_extra_v4.py`)
26 pcap batch yeniden islendi, 5 yeni ozellik icin label=1 vs label=0
ROC-AUC (`sep = max(AUC, 1-AUC)`, 0.5=ayristirma yok, 1.0=mukemmel):

| family | n(1/0) | EN GUCLU yeni sinyal | sep | 2. en guclu | sep | yorum |
|---|---|---|---|---|---|---|
| Wednesday/DoS | 9956/10044 | **SYN Count** (mean 1.55 vs 0.04) | **0.898** | Fwd/Bwd Byte Ratio (ters; med 0.35 vs 49.4) | 0.810 | **COK GUCLU** — MCC=0.9516 dusmez, IYILESEBILIR |
| FTP-Patator | 5983/24017 | **SYN Count** (mean 1.83 vs 0.65) | **0.804** | Down/Up Ratio | 0.582 | ON-TAHMIN YANLIS: RST degil SYN — ama hala COK GUCLU |
| SSH-Patator | 9294/20706 | Down/Up Ratio | 0.655 | SYN Count | 0.643 | ORTA — zaten DR=0.9996/MCC=0.834, ek kazanc kucuk ama zarar yok |
| **WebAttack** | 3556/26444 | **Down/Up Ratio** (ters; med 0.51 vs 0.83) | **0.701** | SYN Count | 0.665 | **YENI BULGU** — plandaki tabloda yoktu, GUCLU |
| Infiltration | 16461/43539 | **SYN Count=0 imzasi** (ters; mean1=0.00 vs mean0=1.38) | **0.832** | Fwd/Bwd Byte Ratio | 0.570 | **YENI BULGU** — port-444 akisi hep "kurulu/devam eden" (SYN yok); ephemeral-bucket (82.6/5.3) ile BIRLESINCE Infiltration DR tahminini 0.6-0.85'in UZERINE cekebilir |
| **Bot** | 2075/27925 | **Fwd/Bwd Byte Ratio** (med 5.68 vs 0.80) | **0.794** | SYN Count | 0.660 | **YENI BULGU, agent'in "zayif Down/Up" tahmininden COK daha guclu** — Bot'un MCC 0.43->0.5-0.6 tahminini YUKARI cekebilir |
| **PortScan** | 9869/20131 | **Fwd/Bwd Byte Ratio** (med 2.00 vs 0.32) | **0.784** | RST Count (mean 0.50 vs 0.02) | 0.743 | **3 GUCLU sinyal birden** (+ Down/Up=0.695) — klasik "SYN->RST" + "fwd-heavy" imzasi DOGRULANDI, ephemeral-port bonus ile BIRLESINCE net KAZANC |
| DDoS | 9906/20094 | FIN Count (mean 0.53 vs 0.13) | 0.667 | SYN Count | 0.587 | en zayif kazanc — FAR=0.30 (dispatcher-esik sorunu, [[project_phase2_dispatcher_tuning]]) muhtemelen AYNEN KALIR |

**Genel sonuc**: 8 ailenin **8'i de** en az 1 yeni ozellikte sep>=0.64 gosteriyor;
6/8 ailede sep>=0.78 (COK GUCLU). En onemli surprizler:
- **RST Count Patator icin DEGIL, PortScan icin** is gorudu (klasik SYN-scan->RST
  yaniti, agent'in PortScan tahminini DOGRULUYOR, Patator tahminini YANLISLIYOR
  — ama Patator zaten SYN Count'tan 0.80 sep aliyor, net etki YINE pozitif).
- **Bot ve WebAttack icin Fwd/Bwd Byte Ratio / Down/Up Ratio**, plandaki tabloda
  hic yoktu ama EN GUCLU yeni sinyal cikti — Bot'un port-shortcut kaybini
  (Bolum 1 EMPIRIK DOGRULAMA tablosundaki %36 FPR riski) telafi etme ihtimali
  agent'in ilk tahmininden (MCC ~0.5-0.6) daha iyi olabilir.
- **Infiltration'in "SYN Count=0" imzasi** (sep=0.832) ephemeral-port bucket
  sinyaliyle (Bolum1, 82.6/5.3) BIRLESTIGINDE, Infiltration DR'inin 0.6-0.85
  araliginin UST UCUNA hatta UZERINE cikma ihtimali var — port-shortcut
  kaybi onceki tahminden daha az hissedilebilir.

Tum bu bulgular **v5.0 GO kararini GUCLENDIRIYOR** — yeni 5 ozellik (32->41)
gercekten davranissal ve TUM ailelerde olculebilir sinyal tasiyor, agent'in
bazi spesifik aile-eslesmeleri (RST->Patator) yanlis olsa da TOPLAM etki
beklenenden DAHA OLUMLU.

---

## 3. Init_Win_bytes_forward/backward (DUSUK ONCELIK)

#2 importance (0.0731), host-fingerprint riski tasir AMA porttan daha az
kritik (OS default window'lari evrensel, tek-pcap'e ozel degil). **v5.0'da
DOKUNMA** (cikarmak performansi dusurur, "birlikte arttir" ile celisir). v5.1'de
bucket-vs-keep A/B yap.

---

## 4. Mutlak-rate vs oran ozellikleri

Mevcut `Flow Bytes/s`/`Flow Packets/s` mutlak ve link-hizina bagimli — ama SILME
(Wednesday DoS hacimsel sinyalini zayiflatir, MCC 0.952 regresyon riski). Bunun
yerine EKLE: `Down/Up Ratio` (Bolum 2) + `Fwd/Bwd Byte Ratio = fwd_sum / max(bwd_sum, 1)`.

**DDoS FAR=0.30:** Phase2B kok-nedeni (memory) FAR'in ~%70'inin LSTM'in temiz
port-80/443'u yuksek skorlamasi oldugunu gosterdi — oran bunu *dogrudan*
cozmez ama dengeli-HTTP vs asimetrik-flood icin EK ayirici verir, FAR'i bir
miktar dusurebilir (garanti degil). **Bot MCC=0.43:** port shortcut'i gidince
tek umut davranissal asimetri; `Down/Up`+`Fwd/Bwd Byte Ratio` en dogrudan
ozellikler, ama tek Ares ornegi oldugundan beklenti olculu (MCC ~0.5-0.6).

---

## 5. Dataset genisletme (Infiltration/Bot ek batch)

Kullanilmayan ayni-oturum adaylari (`_scan_train_candidates.json`):
- Thu/Infiltration: **00691(1286), 00685(905)** — v3/v4'te kullanilmadi
  (v3=00676/00688/00694, v4 +00697/00679/00682, val=00690/00700).
- Fri/Bot: **v3 training = 00476(1488)/00478(351)/00464(236)** (TOP_K=3,
  override yok), val=00460/00470 — dogrulandi, disjoint dosyalar, ayni oturum.

**Marjinal deger:** Finding 1 (tek oturum) duzelmediginden ek batch yeni
cesitlilik getirmez. AMA: feature-fix YOKKEN ~degersiz (model zaten portu
ezberliyor); feature-fix VARKEN DEGERLI (davranissal sinyali daha cok
ornekten ogrenir). **Sira:** once feature-only v5.0, sonra +batch v5.1 — bu
tez icin temiz bir ablation verir.

---

## 6. Onceliklendirilmis yol haritasi

**Maliyet ekonomisi:** Bolum 1+2+4 AYNI re-extraction + AYNI retrain pasinda
yapilirsa toplam maliyet TEK bir v3->v4-olcekli efordur. Hepsini tek
`extract_universal_features_v5` + `streaming_training_dataset_v5.csv` +
`retrain_main_models_v5.py` pasinda birlestir.

### v5.0 MINIMAL (ONERILEN — GO)
1. Bolum 2: Active Max/Min -> SYN/FIN/RST + Down/Up Ratio
2. Bolum 1a: ham port -> 7-kovali one-hot (bucketing, augmentasyon yok)
3. Bolum 4: + Fwd/Bwd Byte Ratio
4. Init_Win'e DOKUNMA, ek batch YOK (v5.1'e birak)

Feature sayisi: 32 - 1(port) + 7(bucket) - 2(Active) + 3(SYN/FIN/RST) + 2(ratio)
= **41**. LSTM footgun nedeniyle `n_feat_lstm=41` ile yeniden egitilmeli,
retrain script'inde shape assert eklenmeli.

**Beklenen net (2x ampirik dogrulamayla revize edildi — Bolum1 port-bucket +
Bolum1/EMPIRIK#2 SYN/FIN/RST/ratio AUC):**
- **Wednesday/DoS**: KORUNUR, muhtemelen IYILESIR — SYN Count (sep=0.898) +
  Fwd/Bwd Byte Ratio (sep=0.810) cok guclu yeni sinyaller, MCC=0.9516'nin
  USTUNE cikma sansi var.
- **FTP-Patator**: KORUNUR + iyilesme — SYN Count sep=0.804 (RST degil ama
  esdeger guclu).
- **SSH-Patator**: KORUNUR — zaten DR=0.9996/MCC=0.834 (tavan yakin), Down/Up+SYN
  (sep~0.65) kucuk ek katki.
- **WebAttack**: KORUNUR + iyilesme — Down/Up Ratio (sep=0.701) plandaki
  tabloda olmayan YENI guclu sinyal.
- **PortScan**: KORUNUR + GUCLU iyilesme — ephemeral-port bonus (Bolum1) +
  Fwd/Bwd Byte Ratio (sep=0.784) + RST Count (sep=0.743) + Down/Up (sep=0.695),
  3 yeni guclu sinyal birden.
- **DDoS**: KORUNUR, kucuk iyilesme (FIN Count sep=0.667) — FAR=0.30 muhtemelen
  AYNEN KALIR (dispatcher-esik sorunu, retrain'le cozulmuyor).
- **Infiltration**: DR DUSER ama port-bucket (82.6%/5.3%, sep~0.7-0.8) + YENI
  "SYN Count=0" imzasi (sep=0.832) BIRLESINCE 0.6-0.85 araliginin UST UCUNA
  (belki uzerine) cikma ihtimali var — 0.97'den duser ama 0.22'ye DONMEZ.
- **Bot**: http-bucket zayifligina (%36 FPR) karsi Fwd/Bwd Byte Ratio
  (sep=0.794, plandaki "zayif Down/Up" tahmininden COK daha guclu) — MCC
  hedefi 0.43'ten 0.5-0.6'ya, belki daha yukariya cekilebilir.

**Genel degerlendirme**: 2. ampirik tur, 1. turun bazi pesimist tahminlerini
(Infiltration, Bot) DAHA DA iyilestirdi ve plandaki bazi aile-ozellik
eslesmelerini (RST->Patator yanlis, RST->PortScan + Fwd/BwdRatio->Bot/WebAttack
dogru/yeni) duzeltti. NET SONUC: v5.0'in "performans VE genellenebilirlik
birlikte" hedefini onceki turdan daha guclu sekilde destekliyor — hicbir
ailede "saf kayip" beklenmiyor, en kotu durum (Infiltration) bile port-bucket+
SYN=0 ikili sinyaliyle kismen telafi ediliyor.

### v5.1 FULL (sonra)
5. Bolum 5: Infiltration/Bot ek batch (00691/00685, varsa Bot icin ek aday)
6. Bolum 3: Init_Win bucket A/B

### GO / NO-GO
- **GO: v5.0 minimal** — tek retrain pasinda hem genelleme (port+olu boyut)
  hem performans (SYN/RST/ratio) ele alindigindan "birlikte arttir" hedefini
  dogrudan karsilar. **2. ampirik tur (EMPIRIK DOGRULAMA #2) bu kararı
  GUCLENDIRDI**: 8/8 aile en az 1 yeni ozellikte sep>=0.64, 6/8 aile
  sep>=0.78 — beklenenden daha genis ve daha guclu kazanc.
- **NO-GO simdilik:** Bolum 1b augmentasyon (one-hot ile gereksiz + yeni
  shortcut), Bolum 3 Init_Win (dusuk kaldirac).
- **Ham veri**: `_check_v5_new_features.py` / `_check_v5_new_features.json`
  (26 batch, v4 egitim kompozisyonuyla AYNI).

---

## Notlar / riskler

- **Footgun (satir 559-564):** Feature sayisi 32->41 olunca LSTM yeni setle
  egitilmezse metrikler sessizce bozulur — retrain script'inde `n_feat_lstm`
  assert'i ekle.
- `retrain_main_models_v4.py` deseni non-destructive (yeni `_v5` suffix) —
  ayni desende devam et: `extract_universal_features_v5`,
  `build_streaming_dataset_v5.py`, `retrain_main_models_v5.py`,
  `pcap_validation_pilot_v5.py`, `validate_all_days.py` (`MODEL_VERSION=v5`).
- Tez cercevesi: v5.0, Finding 2'yi bir "limitasyon itirafi"ndan bir "tasarim
  iyilestirmesi"ne cevirir — Infiltration/Bot'ta beklenen ve durust DR dususu,
  tespiti davranissal temele oturtarak uretim-genellenebilirligini artirir;
  standart-port/hacimsel aileler hem korunur hem yeni ozelliklerden faydalanir.

## v5.0 sonrasi yapilacak dogrulamalar

1. Wednesday 12-batch pilot DR/FAR/MCC, v4 (MCC=0.9516) ile karsilastir —
   port-80 sinyali "http" kovasina ininca Wednesday'in volumetrik+SYN
   sinyallerine kaymasi BEKLENIR, dusmemesi gerek.
2. `validate_all_days.py` (`MODEL_VERSION=v5`) tum 31 batch — 7 aile +
   GENEL tablo, v4 ile yan yana.
3. SYN/RST/Down-Up'in aile-bazli label=1 vs label=0 ayristiriciligi
   (egitim verisinde) — Bolum 2 hipotezini dogrula.
