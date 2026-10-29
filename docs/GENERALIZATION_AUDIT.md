# v4 Pipeline Generalization Audit

**Soru**: v4'ün "7/7 saldırı ailesi >=%90 DR" sonucu gerçekten genellenebilir mi,
yoksa pipeline/feature-engineering kaynaklı bir "kısayol" (shortcut) artefaktı mı?

**Yöntem**: `pcap_validation_pilot.py` (32 raw feature), `build_streaming_dataset_v3.py`
(SCAN_DEFS), `validate_all_days.py` (GROUPS), `retrain_main_models_v4.py`
(preprocessing) kod incelemesi + `MainRF_v4.joblib` feature_importances_ +
`streaming_training_dataset_v4.csv` üzerinde aile-bazlı port/label dağılım analizi.

---

## Finding 1 — Train/val aynı saldırı oturumunu paylaşıyor (7/7 aile)

`build_streaming_dataset_v3.py` (SCAN_DEFS, training GT) ve `validate_all_days.py`
(GROUPS, validation GT) **her aile için BİREBİR AYNI** `make_pair_port_gt(ip_a, ip_b,
ports, family)` tanımını kullanıyor:

| family | IP pair / port (train == val) |
|---|---|
| FTP-Patator | 172.16.0.1 <-> 192.168.10.50 : 21 |
| SSH-Patator | 172.16.0.1 <-> 192.168.10.50 : 22 |
| WebAttack | 172.16.0.1 <-> 192.168.10.50 : 80 |
| Infiltration | 192.168.10.8 <-> 205.174.165.73 : 444 |
| Bot | 205.174.165.73 <-> 192.168.10.x : herhangi port |
| PortScan / DDoS | 172.16.0.1 <-> 192.168.10.50 : herhangi/80 |

**Verdict: (A) DATASET LIMITATION.** CICIDS2017'de her aile için tek bir
saldırgan/kurban IP çifti ve tek bir capture günü/oturumu var — train/val ayrımı
batch dosyası seviyesinde disjoint olsa da, "oturum" seviyesinde örtüşme
kaçınılmaz. Bu literatürde bilinen, citable bir sınırlama (Engelen et al. 2021).
**Tek başına kabul edilebilir** — TEK BAŞINA bu, modelin "IP/port kimliğini"
ezberlemesine yol açmaz, EĞER feature seti IP/port kimliği taşımıyorsa. Ama
Finding 2 bunu tam olarak ihlal ediyor.

---

## Finding 2 — "Destination Port" ham özellik olarak modele veriliyor (TEKNİK SORUN)

`pcap_validation_pilot.py:363`: `"Destination Port": float(flow["dst_port"])` —
32 raw_columns'un **#1'i**, sadece StandardScaler ile normalize edilip RF/LSTM'e
ham sayısal değer olarak veriliyor.

### Kanıt 1 — Feature importance (MainRF_v4, 100 ağaç)

```
 1. Destination Port                    0.0921   <-- #1/32 (uniform ~0.031, ~3x)
 2. Init_Win_bytes_forward              0.0731   <-- #2 (ayrı bir "host-fingerprint" riski)
 3. Max Packet Length                   0.0724
 4. Packet Length Std                   0.0543
 ...
```

### Kanıt 2 — Aile-bazlı `Destination Port` / label ayrışması (`streaming_training_dataset_v4.csv`)

| family | label=1 (attack) dominant port(s) | label=0 top ports | ayrışma | port anlamı |
|---|---|---|---|---|
| DoS (Çarşamba) | 80 (%88.5) | 443, **80 (%14.7)**, 49533 | KISMEN örtüşüyor | standart HTTP |
| FTP-Patator | **21 (%93.2)** | 443, 80, 53 — 21 yok | TAM ayrışma | standart FTP portu |
| SSH-Patator | **22 (%95.2)** | 443, 80, 53 — 22 yok | TAM ayrışma | standart SSH portu |
| WebAttack | 80 (%100) | 443, **80 (%20.1)**, 53 | KISMEN örtüşüyor | standart HTTP |
| **Infiltration** | **54119 (%68.4) + 444 (%17.4) + 54131 (%13.9)** | 443, 80, 53 — bunlar yok | **TAM ayrışma** | **54119/54131 = RASTGELE EPHEMERAL portlar (TEK seferlik, capture'a özel)** |
| Bot | **8080 (%100)** | 80, 443, 53 — 8080 yok | TAM ayrışma | bu özel Ares botnet örneğinin C2 portu |
| PortScan | dağınık (en yüksek port %0.15) | 443 dominant | ayrışma YOK | port kendisi shortcut değil — iyi |
| DDoS | 80 (%96.7) | n=94 (çok küçük örnek) | belirsiz | standart HTTP hedefi |

### Yorum — neden bu önemli

- **FTP-Patator/SSH-Patator/DDoS/WebAttack (port 21/22/80)**: Bu portlar
  *standart servis portları* — "dst_port==21 → FTP saldırısı olabilir" production'da
  da gerçek bir sinyal taşır (anlamlı bir "shortcut", tamamen genellenemez değil).
  Risk **ORTA** seviyede.
- **Infiltration (port 54119/54131)**: Bu portlar **ephemeral/rastgele** —
  işletim sisteminin o TCP bağlantısı için o anda rastgele atadığı portlar.
  Üretimde AYNI infiltration aracı bile çalıştırılsa **farklı** ephemeral
  portlar kullanır. Model `Destination Port ∈ {54119, 444, 54131, ...}` gibi bir
  kural öğrendiyse, bu **literal bu PCAP'e özel bir "flow ID" ezberi** —
  genelleme değeri **SIFIRA YAKIN**. v3→v4'teki DR 0.0044→0.9656 sıçramasının
  büyük kısmı muhtemelen budur: yeni eklenen 3 batch, aynı oturumun
  444/54119/54131 portlu flow'larını train setine taşıdı, validation batch'leri
  de aynı portları içeriyor.
- **Bot (port 8080)**: GT fonksiyonu port-agnostik ("herhangi port") olsa da,
  CICIDS2017'deki bu TEK Ares botnet örneği C2 için sadece port 8080 kullanıyor
  → veri ampirik olarak port-spesifik. Production'da farklı C2 portu kullanan
  bir bot bu modeli muhtemelen atlatır.
- **PortScan**: Beklendiği gibi port feature'ı shortcut DEĞİL (saldırı zaten
  çok sayıda farklı portu tarıyor) — bu ailenin yüksek DR'si daha güvenilir.

**Verdict: (B) TEKNİK SORUN — confirmed.** Finding 1'in "kaçınılmaz aynı oturum"
durumunu, `Destination Port` özelliği **istismar edilebilir bir kısayola**
çeviriyor. Infiltration için bu kombinasyon kritik; Patator/DDoS için daha ılımlı.

---

## Audit #2 — Preprocessing fit scope: TEMİZ (A)

`retrain_main_models_v4.py`: `train_test_split` → `scaler.fit_transform(X_train)`.
Scaler SADECE `streaming_training_dataset_v4.csv` üzerinde fit ediliyor,
validation pcap'leri fit'e karışmıyor. **Sızıntı yok.**

(Yan not: internal split satır-bazlı rastgele, batch/oturum-bazlı değil — yani
retrain script'inin raporladığı "internal MCC=0.99" rakamı kendi içinde optimist,
ama bu rapor edilen per-family validation sayılarını etkilemiyor çünkü onlar
tamamen ayrı pcap batch'lerinden geliyor.)

---

## Audit #3/#4 — Diğer "kimlik benzeri" özellikler ve Friday/Bot-DDoS

- **Init_Win_bytes_forward/backward** (rank #2, #11): TCP initial window size,
  işletim sistemi/TCP stack konfigürasyonuna bağlı — CICIDS2017 testbed'indeki
  SABİT host'lara (192.168.10.50, 172.16.0.1, vb.) özgü bir "host fingerprint"
  olabilir. Port kadar kritik değil ama AYNI mekanizma: production'daki farklı
  host'lar farklı initial window kullanır. **Orta öncelik, v5 kapsamına dahil
  edilebilir.**
- **IAT özellikleri** (Flow/Fwd IAT Mean/Max, rank #12-15): Zamanlama
  davranışsal bir sinyal olduğu için (saldırı araçlarının periyodikliği genel
  bir özellik) bu daha az "kimlik" riski taşıyor — düşük öncelik.
- **Friday/Bot (MCC=0.43) ve Friday/DDoS (FAR=0.30)**: Bot için port
  feature'ının (8080) etkisi yukarıda not edildi — zayıf MCC'nin port-shortcut'tan
  DAHA İYİ olmasının nedeni muhtemelen GT'nin port-agnostik tanımlanmış olması
  (model port'a güvenemediği bazı durumlarda davranışsal özelliklere düşüyor,
  ama yeterli değil). DDoS'un yüksek FAR'ı port'tan bağımsız, dispatcher eşik
  meselesi (zaten `project_phase2_dispatcher_tuning.md`'de not edilmiş).

---

## SONUÇ ve TEZ İÇİN ÖNERİ

### Aile-bazlı genellenebilirlik tahmini

| family | v4 DR | genellenebilirlik tahmini | gerekçe |
|---|---|---|---|
| Tue/FTP-Patator | 0.9977 | **ORTA-YÜKSEK** | port 21 standart + davranışsal sinyal güçlü (brute-force volumetrik) |
| Tue/SSH-Patator | 0.9996 | **ORTA-YÜKSEK** | port 22 standart + aynı sebep |
| Thu/WebAttack | 0.9769 | **ORTA** | port 80 kısmen örtüşüyor, davranışsal özellikler ağırlıkta |
| **Thu/Infiltration** | **0.9708** | **DÜŞÜK** | %82'si ephemeral/non-standart port shortcut'ı — production'da v3'ün 0.0044'üne yakın davranabilir |
| Fri/Bot | 0.9665 | **DÜŞÜK-ORTA** | tek C2 portuna (8080) bağımlı, zaten MCC=0.43 zayıf |
| Fri/PortScan | 0.9961 | **YÜKSEK** | port feature shortcut değil, gerçekten davranışsal (çoklu-port tarama deseni) |
| Fri/DDoS | 0.9936 | **ORTA-YÜKSEK** | port 80 standart hedef + volumetrik sinyal güçlü |

### Limitations bölümü için önerilen paragraf

> "Doğrulama, CICIDS2017'nin her saldırı ailesi için tek bir capture
> oturumu/IP-port çifti içermesi nedeniyle, eğitim ve doğrulama batch'leri
> oturum düzeyinde örtüşmektedir (bilinen bir CICIDS2017 sınırlaması, bkz.
> Engelen et al. 2021). Bu durum, özellik kümesinde yer alan ham
> 'Destination Port' değişkeninin bazı aileler için (özellikle Infiltration,
> ephemeral port 54119/54131 ve Bot, C2 portu 8080) saldırı/normal ayrımını
> kimlik-temelli bir kısayolla yapmasına izin vermiş olabilir. Standart servis
> portlarına (21/22/80) dayanan aileler (Patator, DDoS) için bu durum daha az
> kritiktir çünkü ilgili port-davranış ilişkisi gerçek ortamlarda da geçerlidir.
> Bu nedenle, Infiltration ve Bot için raporlanan tespit oranlarının üretim
> ortamında PortScan/Patator/DDoS'a kıyasla daha düşük genelleme göstermesi
> beklenir."

### v5 için somut, non-destructive düzeltme planı (kullanıcı onayı bekliyor)

1. `pcap_validation_pilot.py`'da `extract_universal_features()` içinde
   `"Destination Port"` özelliğini **kaldır** veya **bucket'la**
   (well-known: {20,21,22,23,25,53,80,443,3389,8080,...} kategorik/one-hot,
   diğerleri "other") — yeni bir `extract_universal_features_v5` fonksiyonu
   olarak, orijinali bozmadan.
2. (Opsiyonel, ikinci öncelik) `Init_Win_bytes_forward/backward` için aynı
   inceleme — host-fingerprint riski.
3. `build_streaming_dataset_v3.py`'nin v5 türevi ile
   `streaming_training_dataset_v5.csv`'yi YENİ feature setiyle yeniden üret
   (mevcut 31 batch + Wednesday 28 batch'in tamamı yeniden feature-extract
   edilmeli — bu en maliyetli adım).
4. `retrain_main_models_v5.py` ile `MainRF_v5`/`MainLSTM_v5` eğit.
5. `validate_all_days.py` (`MODEL_VERSION=v5`) + `pcap_validation_pilot_v5.py`
   ile tüm 31+12 batch'i tekrar doğrula.
6. **Beklenen sonuç**: Infiltration ve Bot DR'lerinde düşüş (muhtemelen
   v3 seviyesine geri dönüş, ~0.22 ve/veya MCC düşüşü), diğer ailelerde
   küçük/ihmal edilebilir değişiklik. Bu düşüş GÖSTERİRSE, Finding 2'nin
   "Infiltration 0.97 büyük ölçüde port-shortcut'tu" tezini DOĞRULAR —
   tez için bu negatif sonuç bile değerlidir (dürüst genellenebilirlik
   analizi).

**Bu adım 3-5, v3→v4 retrain'iyle aynı büyüklükte bir efor (uzun background
run'lar) — kullanıcı onayı olmadan başlatılmadı.**
