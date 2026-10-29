# Literatür / SOTA Karşılaştırması (CICIDS-2017)

Bu belge, "Real-Time Anomaly Detection in Streaming Big Data Pipelines" tez
projesinin v5 sonuçlarını, CICIDS-2017 veri seti üzerinde çalışan bilinen
IDS/anomali-tespit yaklaşımlarıyla karşılaştırır. Amaç, bu projenin
sonuçlarını literatürdeki bağlama oturtmak ve metodolojik farkı (özellikle
değerlendirme protokolü farkını) açıkça ortaya koymaktır.

> **ÖNEMLİ METODOLOJİK NOT:** Aşağıdaki literatür rakamlarının büyük
> çoğunluğu, veri setinin **rastgele (satır-bazlı) eğitim/test bölmesi**
> üzerinde elde edilmiştir. Bu protokol, akış/zamansal sızıntı (flow/temporal
> leakage) nedeniyle **iyimser** sonuçlar üretir (bkz. `KNOWN_LIMITATIONS.md`
> Madde 1 ve aşağıdaki "Metodolojik uyarı" bölümü). Bu projenin birincil
> başlık (headline) rakamları ise **günler-arası (cross-day)** protokolle
> üretilmiştir; bu yüzden ham sayı karşılaştırması adil değildir — yorum
> bölümüne bakınız.

---

## Karşılaştırma tablosu

| # | Yaklaşım / Kaynak | Yıl | Model ailesi | Değerlendirme protokolü | DR/Recall | FAR/FPR | F1 | MCC / Acc |
|---|---|---|---|---|---|---|---|---|
| 1 | LSTM-RNN (Najar & Sukthankar, drone ağları) | 2024 | LSTM-RNN | Rastgele bölme | ~99.70-99.79% | — | ~99.84-99.89% | Acc ~99.84% |
| 2 | Hibrit RF + CNN-LSTM (attention) | 2025 | RF + CNN-LSTM | Rastgele bölme | 99.34% | — | 99.24% | Acc 99.25% (standalone RF Acc 99.93%) |
| 3 | RF / XGBoost / DNN karşılaştırması (ScienceDirect 2025) | 2025 | RF, XGBoost, DNN | Rastgele bölme | — | — | ağırlıklı 99.8% / makro 93.31% | Acc ~99.93% (RF baskın) |
| 4 | Dengesiz-veri MCC benchmark (SCITEPRESS, CIC-IDS2017 kapsamlı analiz) | 2022 | Denetimli (RF baskın) | Rastgele bölme | TPR ~96% | FPR ~2% | — | **MCC ~0.88** |
| 5 | Düşük-frekanslı saldırılar için MCC değerlendirmesi (arXiv 2512.19203) | 2025 | Çoklu (dengeleme + MCC) | Dengeleme + rastgele | — | — | — | MCC odaklı (dengesiz koşulda RF/MCC en güvenilir) |
| 6 | Robust anomaly detection (LOF/OCSVM/MLP/CNN, arXiv 2506.19877) | 2025 | Density/OCSVM/MLP/CNN | Benign-only / novel-attack | DDoS/PortScan/FTP-Patator'da ~mükemmel | LOF orta FPR | — | denetimli yüksek-hacimde güçlü, nadir saldırıda zayıf |
| — | **BU PROJE (v5) — günler-arası (cross-day) pilot** | 2026 | RF (v5) + LSTM (v5) hibrit dispatcher | **Cross-day (Salı/Perş/Cuma/Pzt)** | **DR = 0.9508** | **FAR = 0.0172-0.0210** | precision 0.9409-0.9543 | **MCC = 0.9301-0.9307** |
| — | **BU PROJE (v5) — tek-gün canlı dilim (Wed DoS-Hulk)** | 2026 | aynı | Tek-gün canlı (NiFi->Kafka->process_pcap) | **DR = 0.999** | **FAR = 5.5%** | — | **MCC = 0.916** |

*Tablo notu:* Boş hücreler (—) ilgili kaynağın o metriği raporlamadığını
gösterir. Literatürdeki çoğu çalışma accuracy/F1 raporlar; FAR ve MCC daha
seyrek raporlanır — bu da dengesiz IDS değerlendirmesinde yaygın bir
eksikliktir.

---

## Yorum

**1. Ham sayı karşılaştırması yanıltıcıdır; protokol farkı belirleyicidir.**
Literatürdeki en yüksek rakamlar (Tablo #1-#3: Acc/F1 ~99.2-99.9%) neredeyse
tamamen **rastgele satır-bazlı bölme** ile elde edilmiştir. CICIDS-2017
doğası gereği zamansaldır: saldırılar beş ardışık gün boyunca belirli akışlar
halinde gerçekleşir ve tek bir saldırı akışı (örn. DoS-Hulk, FTP-Patator) çok
sayıda yüksek-korelasyonlu örnek üretir. Rastgele bölme bu akışların bir
kısmını eğitime, bir kısmını teste düşürür (flow/temporal leakage) ve modeli
gerçekte "ezberlediği" örneklerle test eder. Literatür kritiği bunu açıkça
"random splitting methods can create unrealistic performance estimates" ve
"temporal leakage produces overly optimistic performance estimates that fail
to generalize" şeklinde belgeler. Dolayısıyla bu projenin cross-day
**MCC ~0.93**'ü, literatürdeki rastgele-bölme **Acc/F1 ~0.99**'larıyla
DOĞRUDAN kıyaslanmamalıdır — farklı ve daha zorlu bir değerlendirme rejimini
temsil eder.

**2. Adil kıyas: aynı protokoldeki rakamlarla.** Aynı (zorlu) değerlendirme
felsefesini benimseyen çalışmalarla kıyaslandığında bu projenin sonuçları
güçlüdür. Tablo #4'teki kapsamlı CIC-IDS2017 analizinde, dengesiz koşulda en
iyi denetimli modeller **MCC ~0.88, TPR ~96%, FPR ~2%** bildirmektedir. Bu
projenin cross-day rejimindeki **MCC 0.9301-0.9307, DR 0.9508, FAR
0.0172-0.0210** değerleri, bu referansla aynı bandın üst kısmındadır (özellikle
MCC ve FAR açısından), üstelik bu rakamlar **canlı akış (streaming) özellik
çıkarımı** ve **gerçek bir NiFi->Kafka->Python uçtan-uca hattı** üzerinde
üretilmiştir — çoğu literatür çalışması ise hazır (offline, etiketli)
flow-CSV'leri üzerinde batch olarak çalışır.

**3. Metrik seçimi açısından bu proje literatürün doğru tarafındadır.**
Yeni literatür (Tablo #4-#5), dengesiz IDS verisinde accuracy/F1'in tek
başına yanıltıcı olduğunu ve **MCC'nin azınlık saldırılarını gerçekten
yakalayan modelleri ayırt eden en güvenilir metrik** olduğunu vurgular. Bu
proje değerlendirmesinin MCC + FAR + precision/recall üzerine kurulu olması
(yalnızca accuracy'ye güvenmemesi) güncel en iyi pratikle uyumludur.

**4. Mimari konumlandırma.** Literatürdeki en güçlü hatlar ya saf hibrit
derin öğrenme (RF+CNN-LSTM, Tablo #2) ya da ensemble/data-augmentation
yaklaşımlarıdır. Bu proje, bunlardan farklı olarak **saldırı-türüne-özel
(attack-specific) dispatcher** mimarisi kullanır: trafik volumetrik (RF) ve
zamansal (LSTM + ilerlemeli sayaç) dallara yönlendirilir. Bu, "one-size-fits-all"
IDS zayıflığını hedefleyen ve gerçek zamanlı düşük-gecikme kısıtını gözeten,
literatürde göreli olarak az çalışılmış bir tasarım eksenidir.

---

## Metodolojik uyarı (tezde vurgulanması önerilir)

Tez metninde bu karşılaştırma sunulurken, literatürdeki ~%99 rakamlarının
çoğunlukla rastgele bölme kaynaklı iyimser tahminler olduğu açıkça
belirtilmelidir. Bu projenin daha düşük görünen (ama daha gerçekçi)
cross-day MCC ~0.93 değeri bir zayıflık değil, **daha dürüst bir
genellenebilirlik ölçümüdür**. Bu konu `KNOWN_LIMITATIONS.md` Madde 1 ile
birlikte okunmalıdır.

---

## Kaynaklar

- [Improving intrusion detection using LSTM-RNN to protect drones' networks — ScienceDirect (2024)](https://www.sciencedirect.com/science/article/pii/S1110866524000641)
- [Optimizing IDS with Hybrid Random Forest and CNN-LSTM Models — ResearchGate (2025)](https://www.researchgate.net/publication/392520395_Optimizing_Intrusion_Detection_System_IDS_with_Hybrid_Random_Forest_and_CNN-LSTM_Models_for_Improved_Accuracy_and_Efficiency)
- [Enhancing IDS performance: comparative analysis of Random Forest, XGBoost, and Deep Neural Networks — ScienceDirect (2025)](https://www.sciencedirect.com/science/article/pii/S2666827025001215)
- [Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017 — SCITEPRESS (2022)](https://www.scitepress.org/Papers/2022/107740/107740.pdf)
- [Evaluating MCC for Low-Frequency Cyberattack Detection in Imbalanced Intrusion Detection Data — arXiv (2025)](https://arxiv.org/html/2512.19203)
- [Robust Anomaly Detection in Network Traffic: Evaluating ML Models on CICIDS2017 — arXiv (2506.19877)](https://arxiv.org/html/2506.19877)
- [A Deterministic Comparison of Classical ML and Hybrid Deep Representation Models for IDS on NSL-KDD and CICIDS2017 — MDPI Algorithms (2025)](https://www.mdpi.com/1999-4893/18/12/749)
- [Deep Learning for Contextualized NetFlow-Based NIDS: Methods, Data, Evaluation and Deployment — arXiv (2602.05594)](https://arxiv.org/pdf/2602.05594)
