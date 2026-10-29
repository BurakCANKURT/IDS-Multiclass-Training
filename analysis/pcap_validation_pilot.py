# -*- coding: utf-8 -*-
"""
pcap_validation_pilot.py
========================
GERCEK CICIDS-2017 pcap'leri uzerinde offline_test.ipynb dispatcher
mantigini (SYN-burst -> MainRF / aksi halde MainLSTM + Progressive Counter)
calistirip, 5-tuple ground-truth ile karsilastirarak gercek
DR / FAR / MCC / Macro-F1 olcen PILOT betigi.

Neden bu betik:
- canli_trafik_kaydi.csv tek sentetik kayit; bu betik GERCEK pcap'lerle
  ilk uctan-uca dogrulamayi yapar.
- offline_test.ipynb cell 4/6/8/12 mantigi BIRE BIR buraya tasinmistir
  (modeller bir kez yuklenir, dongu icinde tekrar yuklenmez).

Ground truth:
- Wednesday CICIDS-2017 saldirilari TAMAMEN 172.16.0.1 -> 192.168.10.50:80
  (DoS Hulk / slowloris / Slowhttptest). NFStream-Labelled.csv ile dogrulandi.
- Bu yuzden paket-bazli etiket: ip.addr 172.16.0.1 ve 192.168.10.50 arasinda
  ve victim tarafinda port 80 olan paketler ATTACK, digerleri BENIGN.
- NFStream CSV'nin timestamp'i tek degere ('14:28:01') collapse oldugu icin
  zaman-bazli eslemede KULLANILMAZ; bunun yerine pcap'ler dogrudan 5-tuple
  ile etiketlenir (pcap frame.time_epoch full us cozunurluklu, saglam).

Kullanim:
    python pcap_validation_pilot.py
Ciktilar:
    pcap_validation_results.csv   (paket-bazli)
    pcap_validation_summary.json  (batch + genel metrik ozeti)
"""

import os
import sys
import csv
import json
import glob
import subprocess
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
# sklearn predict_proba her cagrida UserWarning basabiliyor; env ile de bastir
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# YOLLAR / KONFIG
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Paths can be overridden with environment variables (see README).
TSHARK = os.environ.get("TSHARK", r"C:\Program Files\Wireshark\tshark.exe")
PCAP_ROOT = os.environ.get("CICIDS_PCAP_ROOT", os.path.join(PROJECT_DIR, "data", "batch_pcaps"))
PCAP_DIR = os.path.join(PCAP_ROOT, "Wednesday-workingHours")

MAIN_RF_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainRF.joblib")
MAIN_LSTM_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainLSTM.keras")
BUNDLE_PATH = os.path.join(PROJECT_DIR, "saved_models", "MainRF__preprocess.joblib")

OUT_CSV = os.path.join(PROJECT_DIR, "results", "pcap_validation_results.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "results", "pcap_validation_summary.json")

# validate_all_days.py threshold-sweep: 0.5->0.4 lifts Tue/Thu/Fri MCC
# 0.4437->0.4830 (FTP-Patator DR 36%->67%) while Wednesday Slowloris/
# Slowhttptest MCC stays ~flat (0.809->0.812 / 0.897->0.891) and FAR rises
# only ~0.3-0.4pts everywhere (project_all_days_validation.md).
ALERT_THRESHOLD = 0.4

# Ground-truth saldirgan/kurban (CICIDS-2017 Wednesday)
ATTACKER_IP = "172.16.0.1"
VICTIM_IP = "192.168.10.50"
VICTIM_PORT = 80

# tshark 16-alan listesi (nifi_csvreader_schema.avsc / canli_trafik_kaydi.csv ile ayni)
TSHARK_FIELDS = [
    "frame.number", "frame.time_epoch", "frame.len", "ip.src", "ip.dst",
    "ip.proto", "ip.hdr_len", "tcp.srcport", "tcp.dstport", "tcp.len",
    "tcp.hdr_len", "udp.srcport", "udp.dstport", "udp.length", "tcp.flags",
    "tcp.window_size_value",
]

# PILOT batch secimi (memory: attack window ~600-1010, clean 0-500 & 1100+)
ATTACK_BATCHES = ["00700", "00750", "00800", "00850", "00900", "01010"]
CLEAN_BATCHES = ["00100", "00200", "00300", "00400", "00500", "01100"]

# Opsiyonel: env ile alt-kume calistir (orn. yalniz clean batch'leri yeniden uretmek)
# PILOT_ATTACK="" PILOT_CLEAN="00100,00200" python pcap_validation_pilot.py
if os.environ.get("PILOT_ATTACK") is not None:
    ATTACK_BATCHES = [b for b in os.environ["PILOT_ATTACK"].split(",") if b]
if os.environ.get("PILOT_CLEAN") is not None:
    CLEAN_BATCHES = [b for b in os.environ["PILOT_CLEAN"].split(",") if b]


# ---------------------------------------------------------------------------
# offline_test.ipynb cell 8 -- YARDIMCI PARSER'LAR (bire bir)
# ---------------------------------------------------------------------------
def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_tcp_flags(value):
    if value is None or value == "":
        return 0
    try:
        if isinstance(value, str):
            s = value.strip()
            return int(s, 16) if s.lower().startswith("0x") else int(s)
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# DISPATCHER ESIKLERI (offline_test.ipynb cell 8/12 ile ayni)
# ---------------------------------------------------------------------------
SYN_BURST_MIN = 100
SYN_RATIO_MIN = 0.5
LSTM_WEIGHT = 0.6
COUNTER_WEIGHT = 0.4

# (R8) ZAMANSAL dal olgunluk-kapısı (process_pcap.py R7/R8 ile birebir).
# Akış olgunlaşmamışken (prog_pkts < PROG_MIN_PKTS) ve Counter henüz kanıt
# üretmemişken (cs < ALERT_THRESHOLD), tek-paket LSTM sıçraması ancak
# LSTM_ALERT_FLOOR'u geçerse ALERT olur. Gerçek DoS'un LSTM bandı >=0.987,
# benign dış-host HTTP/CDN kuyruğu <=0.92 -> 0.95 ikisini ayırır (FP 52->0).
LSTM_ALERT_FLOOR = 0.95
DISPATCH_ALERT_THRESHOLD = 0.4   # offline_test.ipynb cell 12 ile aynı (canlı = 0.4)

# (R12) ERKEN-SALDIRI İSTİSNASI: immature akış (prog_pkts<PROG_MIN_PKTS) pencerede
# >=EARLY_ATTACK_MIN_SUSP suspicious paket (is_syn_only / yarı-açık bağlantı imzası)
# TAŞIYOR ve lstm_score>=EARLY_ATTACK_LSTM ise R8 olgunluk-kapısı ATLANIR. Bu,
# Slowloris/Slowhttptest erken paketlerini (lstm 0.85-0.94 + susp>=1) benign
# CDN/HTTP burst'lerinden (susp=0 veya lstm<0.80) ayırır. Bkz. dispatch_packet
# R12 blok yorumu + _scratch_r12_2d.py kanıtı (Slowloris ATTACK:BENIGN = 48:1).
#
# (R34 -- 2026-08-12) KÖK NEDEN + KONSEY GÜVENCESİ. update_progressive'de
# `suspicious = is_syn_only or (is_psh_small and ...)`; yani HER TCP akışının
# AÇILIŞ SYN'i tek başına susp_count=1 üretir. susp>=1 koşulu bu yüzden
# "yeni bir TCP bağlantısı açıldı" demekle ÖZDEŞ -- taban-oranı 1.0 olan,
# sıfır bilgi taşıyan bir kanıt. Sonuç: lstm>=0.85 olan HER yeni bağlantının
# ilk paketlerinde early_attack=True olup olgunluk-kapısı (ve çok-sınıflı
# hatta ayrıca low_syn_dos_gate + webattack_abs_floor) topluca ATLANIYORDU.
# CANLI KANIT (2026-08-12 23:10-23:28, saldırı yok): 192.168.1.102'den
# port-80'e açılan 7 meşru bağlantının 7'si de ATTACK, risk
# 0.9764616787433624 -- farklı hedef IP'lerde 16 hanede birebir aynı.
# R12'nin dayandığı 48:1 oranı v2-model/pre-R28 dönemine aitti; güncel
# yığında yeniden ölçüldü (validate_slow_attacks korpusu, 12 batch):
#   early_attack & susp==1 -> saldırı 2549 / benign 1308 (~2:1, ayırt edici DEĞİL)
#   early_attack & susp>=2 -> saldırı  737 / benign   10 (~73:1, ayırt edici)
#
# (ÖLÇÜLDÜ VE REDDEDİLDİ) MIN_SUSP 1 -> 2: bu korpusta İYİ (MCC 0.9323->0.9367,
# FAR 0.0100->0.0084, DR -0.0008) AMA çok-sınıflı hatta AĞIR REGRESYON --
# offline_test_multiclass DR 0.9655->0.9205, MCC 0.9730->0.9407, FP kazancı
# SIFIR. Kaybedilen 138 paketin 138'i de GERÇEK DoS-Hulk ve hepsini
# `low_syn_dos_gate` kesiyor (susp=1, immature, vol_syn<20). Yani susp
# eşiğini yükseltmek gerçek saldırıyı da vuruyor: DoS-Hulk immature paketleri
# (lstm p50=0.982) ile meşru yeni bağlantı (lstm 0.9765) bu eksende
# AYIRT EDİLEMİYOR.
#
# (SEÇİLEN ÇÖZÜM) MIN_SUSP=1 KORUNUR; bunun yerine early_attack İSTİSNASI
# ancak konsey (RF+HGB) itiraz ETMİYORSA verilir -- bkz.
# committee_waives_early_attack(). Ayırt edici sinyal ÖLÇÜLDÜ:
#   gerçek DoS-Hulk immature (n=138): HGB P(BENIGN) p50=0.0105, MAKS 0.0756
#   canlı meşru yeni bağlantı (n=12): HGB P(BENIGN) 0.9992-0.99999
# İki dağılım ARASINDA 0.08-0.99 boşluğu var -- tam ayrışma. (RF tek başına
# YETMEZ: aynı iki kümede 0.765-0.864 vs maks 0.9247, iç içe.)
EARLY_ATTACK_LSTM = 0.85
EARLY_ATTACK_MIN_SUSP = 1
# (R34) early_attack MUAFİYETİNİ bloke etmek için konseyin iki üyesinin de
# BENIGN'e vermesi gereken minimum olasılık. 0.5 = "argmax BENIGN" ile
# eşdeğer (ikili başta birebir; çok-sınıflı başta p>=0.5 argmax'ı zaten ima
# eder, yani biraz DAHA muhafazakâr). R26'nın BASTIRMA vetosu (0.95) ile
# KARIŞTIRMA: burada bastırma değil, bir GÜVENLİK KAPISINI ATLAMA hakkı
# veriliyor -- muafiyeti geri çekmek için daha zayıf bir kanıt yeterlidir
# (asimetrik, muhafazakâr yönde).
COMMITTEE_WAIVER_BENIGN_P = 0.5

# (R13) KURULU-ISTEMCI INIT-WINDOW ISTISNASI (Madde-1 FP fix):
# R8/R12 olgunluk-kapisi yalnizca lstm_score<LSTM_ALERT_FLOOR(0.95) ise immature
# akisi bastiriyordu. AMA tek bir ~300B PSH+ACK port-80 paketine (akisin ilk/erken
# ileri paketi) LSTM TEK BASINA 0.98+ "saldiri" guveni verebiliyor (lstm>=FLOOR)
# -> kapidan kaciyor -> FP. canli_trafik_kaydi.csv'de bu, 192.168.10.15'in CDN/
# Google port-80 hedeflerine gonderdigi 3 benign HTTP-GET paketini ATTACK
# skorlatti (satir 347/351/352, risk 0.986-0.994).
# AYIRT EDICI SINYAL (genellenebilir, host/IP-bagimsiz): Init_Win_bytes_forward.
#   - DoS-Hulk/Slowloris/Slowhttptest saldiri araclari (Linux) init window = 29200
#     SABIT (_scratch_madde1_separability.py: 678/678 DoS-Hulk LSTM TP'sinde 29200,
#     std=0; slow-attack SYN paketleri de 29200). Bu, saldiri imzasinin dogal bir
#     parcasi (paket-boyutu/port shortcut'a bagimli DEGIL).
#   - 3 FP benign istemci (Windows tarayici) init window = 251-258 (handshake
#     sonrasi window-scaling ile kucuk ham deger; capture oturum-ortasinda basladigi
#     icin SYN gorulmemis).
# 512 esigi bu iki dagilimi mukemmel ayirir: DoS=29200 >> 512 >> istemci~250.
# KOK NEDEN OZETI: LSTM, port-80 + ~300B PSH+ACK kombinasyonunu (DoS-Hulk HTTP-GET
# flood imzasiyla coklu ortusen) tek pakette saldiri sanir; init_win_fwd bu
# ortusmeyi kiran tek-paket-icinde-mevcut bir akis-durumu sinyalidir.
# NOT (pp<=1 sarti, asagida dispatch_packet'te): "SYN gorulmemis" artefakti
# capture sinirinda SADECE benign istemcilerde degil, pp=2 saldiri paketlerinde
# (iwf=229, Slowloris/Slowhttptest) de olusabiliyor -> client_established
# SADECE flow_table'in pp<=1 (ilk-gorulen) paketlerine uygulanir, bkz.
# dispatch_packet R13 yorumu.
INIT_WIN_CLIENT_MAX = 512

# Port-tarama (scan) imzasi -- bir (src_ip,dst_ip) ikilisinin gordugu FARKLI
# dst_port sayisi. PortScan (Friday) icin is_flood SYN-ratio<0.5 oldugundan
# hicbir zaman tetiklenmiyor (RF-yonlendirilen paketlerin %99'u BENIGN);
# normal istemci-sunucu etkilesimleri tek hanede kalir, nmap tarzi taramalar
# onlarca/yuzlerce farkli portu hizla dener (project_all_days_validation.md).
PORT_SCAN_MIN_PORTS = 40   # bu sayida farkli dst_port -> is_scan=True
PORT_SCAN_FULL_PORTS = 80  # bu sayida farkli portta scan_conf = 1.0

# ---------------------------------------------------------------------------
# PROGRESSIVE COUNTER (yeniden tasarlandi - root cause #3 fix)
# ---------------------------------------------------------------------------
# ESKI tasarim (kumulatif susp + ham-paket-maturity) uzun-omurlu BENIGN
# oturumlari (443/22/21) saldiri gibi isaretliyordu: total_pkts>=40 -> maturity=1,
# ve TLS keepalive/alert gibi "PSH + kucuk payload" paketleri kumulatif susp'i
# kolayca PROG_SUSP_PKT_REF=20'ye dolduruyordu -> counter_score ~1.0 (FP).
# Enstrumantasyon (instrument_score_terms.py) kaniti: clean batch 00635'te
# LSTM+Counter FP'lerinin %92.5'i SADECE counter_score>=0.5'ten (lstm_score<0.5)
# kaynakli, dst_port 443 dominant; ve counter'in GERCEK saldiri (Slowloris/
# Slowhttptest) recall'ina katkisi = 0 (lstm zaten %91-100 yakaliyor).
#
# YENI tasarim:
#  1) "olgunluk = uzun akis" fikri KALDIRILDI. Skor, kumulatif sayaca degil,
#     KAYAN PENCERE (son PROG_WINDOW paket) icindeki suspicious ORANINA dayanir
#     -> uzun benign akista oran dusuk kalir, sonsuz birikim olmaz.
#  2) is_syn_only (yarim-acik / yavas-baglanti imzasi) GUCLU sinyal.
#  3) TLS/sifreli portlarda (443/465/993/995/8443) "PSH + kucuk payload"
#     keepalive/alert kayitlari NORMAL ve yaniltici -> bu sinyal BASTIRILIR;
#     yalnizca is_syn_only sayilir.
#  4) Skor, suspicious orani benign tabani (PROG_BASE_RATE) ASTIKTAN sonra
#     dogrusal yukselir; akis ancak min PROG_MIN_PKTS paketten sonra skor uretir
#     (kucuk akislarda gurultuyu engeller).
PROG_WINDOW = 30          # kayan pencere boyu (son N paket bayragi)
PROG_MIN_PKTS = 12        # bu paketten az akista counter_score = 0 (gurultu engeli)
PROG_BASE_RATE = 0.35     # bu suspicious-oraninin ALTI benign sayilir (skor=0)
PROG_FULL_RATE = 0.85     # bu suspicious-oraninda counter_score = 1.0
TLS_PORTS = frozenset({443, 465, 563, 636, 853, 989, 990, 993, 995, 8443})

# ---------------------------------------------------------------------------
# (R17) PARITE: process_pcap.py'ye eklenen iki dispatcher degisikligi
#   (1) AKIS-YONU (orientation) duzeltmesi -- flow["dst_port"]/["fwd_src_key"]
#       artik "ilk gorulen paketin gonderen tarafi" degil, SYN bayragi ya da
#       efemeral-port araligi ile bulunan ISTEMCI tarafina sabitlenir.
#   (2) YON-DUYARLI KUCUK-PSH kurali -- sabit TLS_PORTS beyaz-listesi yerine
#       "karsi yon hic VERI gonderdi mi?" testi (slow-DoS tek-yonludur, benign
#       keepalive karsiliklidir).
# Bu dosya validate_offline_predictions.py / validate_all_days.py /
# validate_slow_attacks.py / pcap_validation_pilot_v5.py'nin ORTAK dispatcher
# kaynagidir; canli kodla parite olmadan olculen sayilar canli davranisi
# temsil etmez (bkz. [[project-dispatcher-parity-fix-pending-rerun]]).
#
# PILOT_R17=0 ile R17 ONCESI davranis birebir geri getirilir (eski tez
# sayilarini yeniden uretmek / A-B olcum yapmak icin).
# Iki degisiklik AYRI AYRI da kapatilabilir (PILOT_R17_ORIENT / PILOT_R17_PSH),
# boylece bir regresyonun hangi degisiklikten geldigi tek kosuda izole edilir.
_R17_DEFAULT = os.environ.get("PILOT_R17", "1")
R17_ORIENT = os.environ.get("PILOT_R17_ORIENT", _R17_DEFAULT) != "0"
R17_PSH = os.environ.get("PILOT_R17_PSH", _R17_DEFAULT) != "0"
EPHEMERAL_PORT_MIN = 32768
# (R18) R17 orientation'in yan etkisini (uydurma forward yarisi) kapatan kapi;
# ayrintili gerekce dispatch_packet icinde. PILOT_R18=0 ile kapatilabilir.
R18_ORIENT_GATE = os.environ.get("PILOT_R18", "1") != "0"
# (R21) kucuk-PSH kuralina zamanlama (IAT) kapisi -- process_pcap.py ile
# parite, gerekce orada (PSH_SLOW_IAT_US yorumu). PILOT_R21=0 ile kapatilabilir.
R21_PSH_IAT_GATE = os.environ.get("PILOT_R21", "1") != "0"
PSH_SLOW_IAT_US = 300_000  # 0.3s (mikrosaniye)
VOL_WINDOW_SECONDS = 60  # process_pcap.py ile ayni deger (bkz. o dosyadaki R17/R22 yorumlari)
# (R22) self.volumetric (pkts/syn) VE self.scan (port kumesi) icin kayan
# zaman penceresi -- process_pcap.py ile parite (bkz. o dosyadaki R22 yorumu).
# Bu betikte flow_table = flow_table_cls() HER BATCH icin sifirdan kuruluyor
# (validate_all_days.py main dongusu) ve batch'ler ~23sn suruyor -- yani
# VOL_WINDOW_SECONDS=60 penceresi bu harness'te PRATIKTE NO-OP olmali (pencere
# hicbir batch icinde dolmuyor); asagidaki A/B, canli koddaki R22 fix'inin
# offline metrikleri BOZMADIGINI dogrulamak icindir, iyilestirme beklenmez
# (iyilesme yalnizca gunler suren canli kosuda gorulur). PILOT_R22=0 ile
# R22-oncesi (sinirsiz-birikim) davranisi geri getirilir.
R22_SCAN_WINDOW = os.environ.get("PILOT_R22", "1") != "0"

# ── (R33) AKIS-BASI SABIT-PAYDALI PAKET HIZI -- process_pcap.py ile parite ──
# Gerekce/olcum process_pcap.py'deki R33 yorumunda. PILOT_R33=0 ile R33-oncesi
# (rate_value > VOLUMETRIC_THRESHOLD and flow_total_pkts >= 2) davranisi geri
# gelir -- A/B icin.
R33_RATE_WINDOW = os.environ.get("PILOT_R33", "1") != "0"
RATE_WINDOW_SECONDS = 0.1
RATE_TRIGGER_PPS = 25_000.0
RATE_WIN_MIN_PKTS = int(RATE_TRIGGER_PPS * RATE_WINDOW_SECONDS)  # 2500

# ── (R28) HACIMSEL DAL IYILESTIRMELERI -- process_pcap.py ile parite ────────
# Gerekce/olcum process_pcap.py'de (SYN_ABS_MIN / SCAN_RF_VETO_MAX_PORTS /
# CONN_FLOOD_MIN yorumlari). Her biri AYRI AYRI kapatilabilir ki bir
# regresyonun hangi maddeden geldigi tek kosuda izole edilebilsin.
#   PILOT_R28_SYNABS=0  -> O1.1 (mutlak-SYN kolu) kapali
#   PILOT_R28_SCANVETO=0-> O1.2 (scan imzasina RF vetosu) kapali
#   PILOT_R28_CONNFLOOD=0->O1.3 (hedef-toplulastirmali baglanti seli) kapali
_R28_DEFAULT = os.environ.get("PILOT_R28", "1")
R28_SYN_ABS = os.environ.get("PILOT_R28_SYNABS", _R28_DEFAULT) != "0"
R28_SCAN_RF_VETO = os.environ.get("PILOT_R28_SCANVETO", _R28_DEFAULT) != "0"
R28_CONN_FLOOD = os.environ.get("PILOT_R28_CONNFLOOD", _R28_DEFAULT) != "0"
SYN_ABS_MIN = 150
SYN_ABS_FULL = 300
SCAN_RF_VETO_MAX_PORTS = 150
CONN_FLOOD_MIN = 200
CONN_FLOOD_FULL = 400
# (R28/O2.1) davranissal-dal konsey vetosu (RF+HGB) -- process_pcap.py ile
# parite. PILOT_R28_COMMITTEE=0 ile kapatilabilir (A/B icin).
R28_COMMITTEE = os.environ.get("PILOT_R28_COMMITTEE", _R28_DEFAULT) != "0"
COMMITTEE_VETO_THRESHOLD = 0.95

# ── (R28/Ö2.3) "KAYNAK AÇLIĞI" (starvation) TERİMİ: ÖLÇÜLDÜ, REDDEDİLDİ ─────
# ÖNERİ: Progressive Counter'a paket-sayısından BAĞIMSIZ, süre tabanlı bir
# terim eklemek -- istemci yönünde (a) uzun süredir açık, (b) gerçekten veri
# göndermiş, (c) ortalama ileri-yön yükü çok düşük, (d) karşı taraf hiç veri
# göndermemiş bir akışı slow-DoS saymak. Yapısal gerekçe makul görünüyordu:
# "bağlantıyı açık tutan taraf tanım gereği istemcidir".
#
# GERÇEK VERİDE TERS ÇALIŞIYOR. Wednesday Slowloris/Slowhttptest/GAP
# batch'lerinde, kapının ön-koşullarını sağlayan FORWARD paketler içinde
# (süre>=10s VE ileri-yön yük<=100 B/s) kapısının tetiklenme oranı:
#
#     batch grubu            ATTACK    BENIGN
#     Slowloris               %2.0      %32.7
#     Slowhttptest            %6.2      %57.0
#     GAP_clean (saldırı YOK)   --       %27.2
#
# Yani terim benign trafikte saldırıdan 10-25 KAT fazla tetikleniyor: çok
# sayıda MEŞRU istemci akışı da uzun ömürlü, tek yönlü ve düşük hızlıdır
# (idle keepalive, long-poll, yoklama istemcileri). Ölçülen Slowloris saldırı
# akışlarının p50 süresi zaten yalnızca 1.15s -- "slow" olan şey AKIŞ SÜRESİ
# değil PAKET-ARASI ZAMANLAMA'dır ve onu R21'in IAT kapısı ZATEN kullanıyor.
# KARAR: eklenmedi. Tekrar denenmeden önce bu ölçümü tekrarlayın.


class UniversalRealTimeFlowTable:
    """offline_test.ipynb cell 8 sinifinin bire bir kopyasi."""

    def __init__(self):
        self.flows = {}
        self.volumetric = {}
        # Port-tarama tespiti icin (src_ip, dst_ip) bazli, gorulen FARKLI
        # dst_port kumesi.
        self.scan = {}
        # (R28/O1.3) (dst_ip, dst_port) -> baglanti baslatan FARKLI
        # (src_ip, src_port) kumesi -- process_pcap.py ile parite.
        self.conn_flood = {}

    def update_volumetric(self, packet):
        src_ip = packet.get("ip.src", "0.0.0.0")
        dst_ip = packet.get("ip.dst", "0.0.0.0")
        dst_port = _to_int(packet.get("tcp.dstport") or packet.get("udp.dstport"), 0)
        flags_int = _parse_tcp_flags(packet.get("tcp.flags"))
        now_s = _to_float(packet.get("frame.time_epoch"), 0.0)

        key = (src_ip, dst_ip, dst_port)
        st = self.volumetric.get(key)
        if st is None:
            st = {"pkts": 0, "syn": 0, "win_start": now_s}
            self.volumetric[key] = st
        if R22_SCAN_WINDOW and now_s > 0 and (now_s - st.get("win_start", now_s)) > VOL_WINDOW_SECONDS:
            st["pkts"] = 0
            st["syn"] = 0
            st["win_start"] = now_s
        st["pkts"] += 1
        if (flags_int & 0x02) and not (flags_int & 0x10):
            st["syn"] += 1

        syn_ratio = st["syn"] / st["pkts"] if st["pkts"] else 0.0
        # (R28/O1.1) iki kollu flood imzasi -- process_pcap.py ile parite.
        is_flood = st["syn"] >= SYN_BURST_MIN and syn_ratio >= SYN_RATIO_MIN
        if R28_SYN_ABS and st["syn"] >= SYN_ABS_MIN:
            is_flood = True

        # (R28/O1.3) hedef-toplulastirmali baglanti seli -- process_pcap.py ile
        # parite (gerekce orada, CONN_FLOOD_MIN yorumu).
        # SAYAC HER ZAMAN islenir (PILOT_R28_CONNFLOOD=0 olsa bile) -- boylece
        # "yalniz-olcum" modunda distinct_conns CSV'ye yazilip esik GERCEK
        # veriyle kalibre edilebilir; yalnizca KARAR (is_conn_flood) toggle'a
        # baglidir. Ilk denemede esik veriye bakilmadan secilmis ve SSH-Patator
        # (yuzlerce mesru SSH baglantisi acar) hacimsel dala kacip DR'yi
        # 0.947 -> 0.13'e dusurmustu; bu ayrim o hatanin tekrarini onler.
        cf_key = (dst_ip, dst_port)
        cst = self.conn_flood.get(cf_key)
        if cst is None:
            cst = {"conns": set(), "win_start": now_s}
            self.conn_flood[cf_key] = cst
        if R22_SCAN_WINDOW and now_s > 0 and (now_s - cst.get("win_start", now_s)) > VOL_WINDOW_SECONDS:
            cst["conns"] = set()
            cst["win_start"] = now_s
        if (flags_int & 0x02) and not (flags_int & 0x10):
            _sp_cf = _to_int(packet.get("tcp.srcport") or packet.get("udp.srcport"), 0)
            cst["conns"].add((src_ip, _sp_cf))
        distinct_conns = len(cst["conns"])
        is_conn_flood = bool(R28_CONN_FLOOD and distinct_conns >= CONN_FLOOD_MIN)

        scan_key = (src_ip, dst_ip)
        sst = self.scan.get(scan_key)
        if sst is None:
            sst = {"ports": set(), "win_start": now_s}
            self.scan[scan_key] = sst
        if R22_SCAN_WINDOW and now_s > 0 and (now_s - sst.get("win_start", now_s)) > VOL_WINDOW_SECONDS:
            sst["ports"] = set()
            sst["win_start"] = now_s
        ports = sst["ports"]
        # (R11 PARITE KASITLI ATLANDI -- SLOW-ATTACK DR REGRESYONU)
        # process_pcap.py update_volumetric (satir ~344-346) R5 SYN-only filtresi
        # icerir: yalnizca SYN-only paketin dst_port'u "taranan port" sayilir.
        # O filtreyi BURAYA da port etmek offline-canli paritesini saglardi, AMA
        # validate_slow_attacks.py ile olculen agir DR regresyonu yaratti:
        #   Slowloris    DR 0.9234 -> 0.555  (TP 4765->2864, FN 395->2296)
        #   Slowhttptest DR 0.9775 -> 0.9136 (TP 12972->12124, FN 298->1146)
        # KOK NEDEN: Slowloris kurbanin 80 portuna ONLARCA yarim-acik baglanti
        # acar. SALDIRI yonu (att->vic) tek port (80) -> is_scan ZATEN False.
        # ANCAK YANIT yonu (vic->att) paketlerinin dst_port'u = saldirganin her
        # baglantida DEGISEN ephemeral kaynak portudur -> kosulsuz ports.add ile
        # (vic,att) cifti 206 "farkli port" gorur -> is_scan=True olur ve slow
        # saldiri paketleri RF (volumetrik) dalinda yuksek skorla yakalanir.
        # SYN-only filtresi bu yaniti eler (yanit SYN-ACK/ACK, SYN-only degil) ->
        # is_scan=False -> slow paketler LSTM dalina kayar, LSTM bunlari zayif
        # skorlar -> FN. YANI offline slow-DR'si, GERCEKTE port-tarama OLMAYAN
        # (R5'in dogru sekilde elemek istedigi) bir is_scan sinyaline KAZARA
        # bagimliymis. KARAR: parite faydasi (offline-canli olcum uyumu)
        # slow-attack tespit kaybina DEGMEZ -> R5 filtresi PORT EDILMEDI.
        # canli_trafik_kaydi.csv FP'leri bu filtreden BAGIMSIZ (R9 DNS-devolumetric
        # DNS FP'lerini zaten eler; bu CSV'de slow-attack ground-truth YOK).
        # AYRI BULGU (R12 adayi): process_pcap.py CANLI kod R5-filtreli oldugundan
        # CANLI SISTEM slow saldirilari is_scan ile YAKALAMAZ; canli slow tespiti
        # yalnizca LSTM+Counter (update_progressive olgunlugu) dalina dayanir ->
        # ayri dogrulama/iyilestirme gerektirir (bkz. [[project-slow-attack-validation]]).
        ports.add(dst_port)
        distinct_ports = len(ports)
        is_scan = distinct_ports >= PORT_SCAN_MIN_PORTS

        return (is_flood, st["pkts"], st["syn"], syn_ratio, is_scan, distinct_ports,
                is_conn_flood, distinct_conns)

    def update_rate_window(self, flow_key, now_s):
        """(R33) Akis-basi SABIT-PAYDALI kayan pencere paket sayaci --
        process_pcap.py ile parite (gerekce orada). Desen update_volumetric'in
        win_start sifirlamasiyla BIREBIR aynidir."""
        f = self.flows.get(flow_key)
        if f is None:
            return 0
        st = f.get("rate_win")
        if st is None:
            st = {"pkts": 0, "win_start": now_s}
            f["rate_win"] = st
        if now_s > 0 and (now_s - st["win_start"]) > RATE_WINDOW_SECONDS:
            st["pkts"] = 0
            st["win_start"] = now_s
        st["pkts"] += 1
        return st["pkts"]

    def update_progressive(self, packet, flow):
        """Kayan-pencere suspicious-ORANI tabanli progressive counter.

        Donus: (counter_score, susp_window_count, total_pkts, susp_ratio)
        (susp artik kumulatif degil; donen ikinci deger pencere icindeki
         suspicious paket sayisi -- geriye uyumlu imza icin korunur.)
        """
        from collections import deque
        flags_int = _parse_tcp_flags(packet.get("tcp.flags"))
        payload = _to_float(packet.get("tcp.len"), 0.0)
        dst_port = _to_int(flow.get("dst_port"), 0)

        st = flow.get("_prog")
        if st is None:
            st = {"win": deque(maxlen=PROG_WINDOW)}
            flow["_prog"] = st

        is_syn_only = bool((flags_int & 0x02) and not (flags_int & 0x10))
        is_psh_small = bool((flags_int & 0x08) and payload <= 200.0)

        if R17_PSH:
            # (R17) YON-DUYARLI KUCUK-PSH -- process_pcap.py ile parite.
            # Kucuk-PSH yalnizca KARSI YON henuz hic payload gondermemisse
            # suspicious sayilir: slow-DoS tek-yonlu sizintidir, benign
            # keepalive/interaktif trafik karsilikli konusmadir. Port
            # bagimsizdir -> TLS_PORTS beyaz-listesi gereksizlesir.
            src_key_now = f"{packet.get('ip.src', '0.0.0.0')}:" \
                          f"{_to_int(packet.get('tcp.srcport') or packet.get('udp.srcport'), 0)}"
            is_fwd_now = (src_key_now == flow.get("fwd_src_key"))
            if is_fwd_now:
                peer_data_pkts = flow.get("bwd_data_pkts", 0)
            else:
                peer_data_pkts = flow.get("fwd_data_pkts", 0)
            if R21_PSH_IAT_GATE:
                # (R21) bkz. process_pcap.py PSH_SLOW_IAT_US yorumu -- kucuk-PSH
                # ancak KENDI yonundeki en son IAT yavassa (Slowloris/Slowhttptest
                # imzasi) suspicious sayilir; meshru hizli tek-yonlu streaming
                # (ör. HTTP/2 SSE) bu payload bandinda ORTUSUYOR ama zamanlamada
                # 1000x+ farkli.
                _own_iat_list = flow.get("fwd_iat_list") if is_fwd_now else flow.get("bwd_iat_list")
                _own_iat = _own_iat_list[-1] if _own_iat_list else None
                is_psh_small = is_psh_small and (_own_iat is not None) and (_own_iat >= PSH_SLOW_IAT_US)
            suspicious = is_syn_only or (is_psh_small and peer_data_pkts == 0)
        elif dst_port in TLS_PORTS:
            # (R17 ONCESI) TLS/sifreli portlarda PSH+kucuk-payload NORMAL ve
            # yaniltici -> bastir; yalnizca is_syn_only suspicious sayilir.
            suspicious = is_syn_only
        else:
            suspicious = is_syn_only or is_psh_small

        st["win"].append(1 if suspicious else 0)

        total_pkts = flow["fwd_packets"] + flow["bwd_packets"]
        win = st["win"]
        win_len = len(win)
        susp_count = sum(win)

        # (R28/O2.3) KAYNAK ACLIGI TERIMI -- process_pcap.py ile parite.
        # Olgunluk erken-donusunden ONCE hesaplanir (terimin amaci paket-sayisi
        # Kucuk akislarda gurultuyu engelle: yeterli kanit yoksa skor=0.
        if total_pkts < PROG_MIN_PKTS or win_len < PROG_MIN_PKTS:
            return 0.0, susp_count, total_pkts, (susp_count / win_len if win_len else 0.0)

        susp_ratio = susp_count / win_len

        # Benign tabani (PROG_BASE_RATE) altinda skor=0; PROG_FULL_RATE'te 1.0.
        if susp_ratio <= PROG_BASE_RATE:
            counter_score = 0.0
        else:
            counter_score = (susp_ratio - PROG_BASE_RATE) / (PROG_FULL_RATE - PROG_BASE_RATE)
        counter_score = float(max(0.0, min(counter_score, 1.0)))
        return counter_score, susp_count, total_pkts, susp_ratio

    def extract_universal_features(self, packet):
        try:
            src_ip = packet.get("ip.src", "0.0.0.0")
            dst_ip = packet.get("ip.dst", "0.0.0.0")
            src_port = _to_int(packet.get("tcp.srcport") or packet.get("udp.srcport"), 0)
            dst_port = _to_int(packet.get("tcp.dstport") or packet.get("udp.dstport"), 0)
            proto = _to_int(packet.get("ip.proto"), 6)

            flow_key = tuple(sorted([f"{src_ip}:{src_port}", f"{dst_ip}:{dst_port}"])) + (proto,)
            packet_len = _to_float(packet.get("frame.len"), 0.0)
            tcp_payload_len = _to_float(packet.get("tcp.len"), 0.0)
            time_epoch = _to_float(packet.get("frame.time_epoch"), 0.0) * 1_000_000

            src_key = f"{src_ip}:{src_port}"
            dst_key = f"{dst_ip}:{dst_port}"

            if flow_key not in self.flows:
                # (R17) AKIS-YONU (orientation) -- process_pcap.py ile parite.
                # "forward" = akisin ISTEMCI tarafi. SYN varsa kesin sinyaldir;
                # yoksa (mid-stream giris) efemeral-port araligi ile tahmin
                # edilir. Dosyadan tam capture'da akislarin ~%99'u SYN ile
                # basladigi icin CICIDS korpusunda pratikte no-op'tur.
                flip = False
                if R17_ORIENT:
                    _fi = _parse_tcp_flags(packet.get("tcp.flags"))
                    _syn_only = bool((_fi & 0x02) and not (_fi & 0x10))
                    _syn_ack = bool((_fi & 0x02) and (_fi & 0x10))
                    if _syn_only:
                        flip = False
                    elif _syn_ack:
                        flip = True
                    else:
                        dst_eph = dst_port >= EPHEMERAL_PORT_MIN
                        src_eph = src_port >= EPHEMERAL_PORT_MIN
                        if dst_eph != src_eph:
                            flip = dst_eph
                        else:
                            flip = src_port < dst_port

                self.flows[flow_key] = {
                    "start_time": time_epoch, "last_time": time_epoch,
                    # (R17) flipped: akis mid-stream yakalandi ve yon SYN yerine
                    # port sezgiseliyle TAHMIN edildi (R13/R18 kapilari kullanir).
                    "flipped": flip,
                    "fwd_src_key": dst_key if flip else src_key,
                    "fwd_packets": 0, "bwd_packets": 0,
                    "fwd_lengths": [], "bwd_lengths": [],
                    "dst_port": src_port if flip else dst_port,
                    "iat_list": [], "fwd_iat_list": [], "bwd_iat_list": [],
                    "last_fwd_time": None, "last_bwd_time": None,
                    "psh_count": 0, "ack_count": 0,
                    "init_win_fwd": None, "init_win_bwd": None,
                    # (R17) bwd_data_pkts yalnizca update_progressive'in
                    # yon-duyarli kuralindaki sayacdir; oznitelige GIRMEZ.
                    "fwd_data_pkts": 0, "bwd_data_pkts": 0,
                    # (R28/O2.3) ileri yon kumulatif TCP yuku (sayac, oznitelik DEGIL)
                    "fwd_payload_bytes": 0.0,
                }

            flow = self.flows[flow_key]
            diff = time_epoch - flow["last_time"]
            if diff > 0:
                flow["iat_list"].append(diff)
            flow["last_time"] = time_epoch
            if len(flow["iat_list"]) > 50:
                flow["iat_list"].pop(0)

            is_fwd = (src_key == flow["fwd_src_key"])

            if is_fwd:
                if flow["last_fwd_time"] is not None:
                    fdiff = time_epoch - flow["last_fwd_time"]
                    if fdiff > 0:
                        flow["fwd_iat_list"].append(fdiff)
                        if len(flow["fwd_iat_list"]) > 50:
                            flow["fwd_iat_list"].pop(0)
                flow["last_fwd_time"] = time_epoch
            else:
                if flow["last_bwd_time"] is not None:
                    bdiff = time_epoch - flow["last_bwd_time"]
                    if bdiff > 0:
                        flow["bwd_iat_list"].append(bdiff)
                        if len(flow["bwd_iat_list"]) > 50:
                            flow["bwd_iat_list"].pop(0)
                flow["last_bwd_time"] = time_epoch

            flags_int = _parse_tcp_flags(packet.get("tcp.flags"))
            if flags_int & 0x08:
                flow["psh_count"] += 1
            if flags_int & 0x10:
                flow["ack_count"] += 1

            win_size = _to_float(packet.get("tcp.window_size_value") or packet.get("tcp.window_size"), 8192.0)
            if win_size <= 0:
                win_size = 8192.0

            if is_fwd:
                flow["fwd_packets"] += 1
                flow["fwd_lengths"].append(packet_len)
                if flow["init_win_fwd"] is None:
                    flow["init_win_fwd"] = win_size
                if tcp_payload_len > 0:
                    flow["fwd_data_pkts"] += 1
                # (R28/O2.3) starvation terimi icin kumulatif ileri-yon yuku
                flow["fwd_payload_bytes"] = flow.get("fwd_payload_bytes", 0.0) + tcp_payload_len
            else:
                flow["bwd_packets"] += 1
                flow["bwd_lengths"].append(packet_len)
                if flow["init_win_bwd"] is None:
                    flow["init_win_bwd"] = win_size
                # (R17) yon-duyarli kucuk-PSH kurali icin (bkz. update_progressive)
                if tcp_payload_len > 0:
                    flow["bwd_data_pkts"] = flow.get("bwd_data_pkts", 0) + 1

            fwd_sum = sum(flow["fwd_lengths"])
            bwd_sum = sum(flow["bwd_lengths"])
            total_pkts = flow["fwd_packets"] + flow["bwd_packets"]

            fwd_mean = fwd_sum / len(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0
            bwd_len = len(flow["bwd_lengths"])
            bwd_mean = bwd_sum / bwd_len if bwd_len > 0 else 0
            bwd_max = max(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0
            bwd_min = min(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0
            bwd_std = float(np.std(flow["bwd_lengths"])) if bwd_len > 1 else 0.0

            all_lengths = flow["fwd_lengths"] + flow["bwd_lengths"]
            max_pkt_len = max(all_lengths) if all_lengths else 0
            mean_pkt_len = sum(all_lengths) / total_pkts if total_pkts > 0 else 0
            pkt_len_std = float(np.std(all_lengths)) if total_pkts > 1 else 0.0
            pkt_len_var = float(np.var(all_lengths)) if total_pkts > 1 else 0.0

            flow_duration_sec = max((time_epoch - flow["start_time"]) / 1_000_000.0, 1e-6)
            flow_bytes_per_s = float((fwd_sum + bwd_sum) / flow_duration_sec)
            flow_packets_per_s = float(total_pkts / flow_duration_sec)
            bwd_packets_per_s = float(bwd_len / flow_duration_sec)

            master_features = {
                "Destination Port": float(flow["dst_port"]),
                "Total Length of Fwd Packets": float(fwd_sum),
                "Fwd Packet Length Max": float(max(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0),
                "Fwd Packet Length Mean": float(fwd_mean),
                "Bwd Packet Length Min": float(bwd_min),
                "Bwd Packet Length Max": float(bwd_max),
                "Bwd Packet Length Mean": float(bwd_mean),
                "Bwd Packet Length Std": bwd_std,
                "Packet Length Mean": float(mean_pkt_len),
                "Packet Length Std": float(pkt_len_std),
                "Packet Length Variance": float(pkt_len_var),
                "Average Packet Size": float(sum(all_lengths) / total_pkts if total_pkts > 0 else 0),
                "Avg Fwd Segment Size": float(fwd_mean),
                "Avg Bwd Segment Size": float(bwd_mean),
                "Subflow Fwd Bytes": float(fwd_sum),
                "Flow IAT Mean": float(np.mean(flow["iat_list"])) if flow["iat_list"] else 3.0,
                "Fwd IAT Max": float(max(flow["fwd_iat_list"]) if flow["fwd_iat_list"] else 0),
                "Bwd IAT Total": float(sum(flow["bwd_iat_list"])),
                "Bwd IAT Max": float(max(flow["bwd_iat_list"]) if flow["bwd_iat_list"] else 0),
                "Bwd Packets/s": bwd_packets_per_s,
                "PSH Flag Count": float(flow["psh_count"]),
                "ACK Flag Count": float(flow["ack_count"]),
                "Max Packet Length": float(max_pkt_len),
                "Init_Win_bytes_forward": float(flow["init_win_fwd"] if flow["init_win_fwd"] is not None else win_size),
                "Init_Win_bytes_backward": float(flow["init_win_bwd"] if flow["init_win_bwd"] is not None else 0.0),
                "act_data_pkt_fwd": float(flow["fwd_data_pkts"]),
                "Subflow Bwd Packets": float(bwd_len),
                "Active Max": 0.0, "Active Min": 0.0,
                "Flow IAT Max": float(max(flow["iat_list"]) if flow["iat_list"] else 0),
                "Flow Bytes/s": flow_bytes_per_s,
                "Flow Packets/s": flow_packets_per_s,
            }
            return master_features, flow_key
        except Exception:
            return None, None


# ---------------------------------------------------------------------------
# offline_test.ipynb cell 4 -- DINAMIK LSTM YUKLEYICI (bire bir)
# ---------------------------------------------------------------------------
def build_and_load_dynamic_lstm(keras_path):
    import io
    import zipfile
    import h5py
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

    extracted_matrices = {}
    with zipfile.ZipFile(keras_path, "r") as z:
        h5_file_name = next((name for name in z.namelist() if name.endswith(".h5")), None)
        with z.open(h5_file_name) as f:
            file_data = io.BytesIO(f.read())
            with h5py.File(file_data, "r") as h5f:
                def visitor(name, node):
                    if isinstance(node, h5py.Dataset):
                        if "optimizer" not in name and "metrics" not in name:
                            extracted_matrices[name.lower()] = node[()]
                h5f.visititems(visitor)

    lstm_k = lstm_r = lstm_b = dense_k = dense_b = None
    for name, mat in extracted_matrices.items():
        if "lstm" in name or "rnn" in name:
            if "recurrent" in name:
                lstm_r = mat
            elif "bias" in name:
                lstm_b = mat
            elif "kernel" in name:
                lstm_k = mat
        elif "dense" in name or "output" in name:
            if "bias" in name:
                dense_b = mat
            elif "kernel" in name:
                dense_k = mat

    if not all(x is not None for x in [lstm_k, lstm_r, lstm_b, dense_k, dense_b]):
        mats = list(extracted_matrices.values())
        mats_1d = [m for m in mats if len(m.shape) == 1]
        mats_2d = [m for m in mats if len(m.shape) == 2]
        lstm_b = max(mats_1d, key=lambda x: x.shape[0])
        dense_b = min(mats_1d, key=lambda x: x.shape[0])
        lstm_2d = [m for m in mats_2d if m.shape[1] == lstm_b.shape[0]]
        dense_k = next(m for m in mats_2d if m.shape[1] == dense_b.shape[0])
        units_calc = lstm_b.shape[0] // 4
        if lstm_2d[0].shape[0] == units_calc and lstm_2d[1].shape[0] != units_calc:
            lstm_r, lstm_k = lstm_2d[0], lstm_2d[1]
        elif lstm_2d[1].shape[0] == units_calc and lstm_2d[0].shape[0] != units_calc:
            lstm_r, lstm_k = lstm_2d[1], lstm_2d[0]
        else:
            lstm_k, lstm_r = lstm_2d[0], lstm_2d[1]

    n_feat = lstm_k.shape[0]
    units = lstm_r.shape[0]
    n_classes = dense_k.shape[1]
    output_activation = "sigmoid" if n_classes == 1 else "softmax"

    model = Sequential([
        Input(shape=(1, n_feat)),
        LSTM(units),
        Dropout(0.2),
        Dense(n_classes, activation=output_activation),
    ])
    model.set_weights([lstm_k, lstm_r, lstm_b, dense_k, dense_b])
    return model


# ---------------------------------------------------------------------------
# MODEL YUKLEME (BIR KEZ)
# ---------------------------------------------------------------------------
def load_models():
    import joblib
    print("[*] Modeller yukleniyor (bir kez)...")
    bundle = joblib.load(BUNDLE_PATH)
    cfg = {
        "PIPELINE": bundle["pipeline"],
        "RAW_COLUMNS": bundle["raw_columns"],
        "CLASSES_LIST": bundle.get("classes", ["BENIGN"]),
        "VOLUMETRIC_FEATURE": bundle.get("volumetric_feature", "Flow Packets/s"),
        "VOLUMETRIC_THRESHOLD": bundle.get("volumetric_threshold", float("inf")),
    }
    cl = cfg["CLASSES_LIST"]
    cfg["BENIGN_INDEX"] = bundle.get("benign_index", cl.index("BENIGN") if "BENIGN" in cl else 0)

    main_rf = joblib.load(MAIN_RF_PATH)
    if isinstance(main_rf, dict) and "model" in main_rf:
        main_rf = main_rf["model"]
    cfg["main_rf"] = main_rf
    cfg["RF_CLASSES"] = list(getattr(main_rf, "classes_", [0, 1]))
    cfg["RF_BENIGN_POS"] = cfg["RF_CLASSES"].index(0) if 0 in cfg["RF_CLASSES"] else 0

    # PERF (Strateji-1 / compute-budget-latency): RF.predict_proba tek-satir
    # cagrilarda ~33-36ms ile darbogaz. Ayni RF'nin ONNX surumu (MainRF_v5.onnx)
    # ONNX Runtime ile ~0.04ms -> ~900x hizlanma. Cikti birebir korunur: bkz.
    # rf_predict_proba(). ONNX yuklenemezse joblib'e dusulur (fallback).
    cfg["rf_onnx"] = _try_load_rf_onnx(MAIN_RF_PATH)

    cfg["main_lstm"] = build_and_load_dynamic_lstm(MAIN_LSTM_PATH)

    # PERF: model.predict()/eager model(x) tek-satir cagrilarda asiri overhead
    # tasiyor (60k cagri -> dakikalarca). Modeli SABIT input-signature'li bir
    # tf.function'a sararsak grafik BIR KEZ trace edilir ve sonraki tum cagrilar
    # derlenmis grafigi kullanir. Cikti DEGERLERI birebir aynidir (ayni agirliklar,
    # ayni forward pass) -- yalnizca cagri mekanigi hizlanir.
    import tensorflow as tf
    n_feat_lstm = cfg["main_lstm"].input_shape[-1]

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1, n_feat_lstm], dtype=tf.float32)])
    def _lstm_fast(x):
        return cfg["main_lstm"](x, training=False)

    cfg["lstm_fast"] = _lstm_fast
    cfg["n_feat_lstm"] = n_feat_lstm

    print(f"    RF classes_={cfg['RF_CLASSES']} | vol_thr={cfg['VOLUMETRIC_THRESHOLD']} "
          f"| BENIGN_INDEX={cfg['BENIGN_INDEX']} | raw_cols={len(cfg['RAW_COLUMNS'])} "
          f"| lstm_in={n_feat_lstm}")
    return cfg


# ---------------------------------------------------------------------------
# ONNX RF HIZLANDIRMA (Strateji-1): joblib RF.predict_proba -> ONNX Runtime
# ---------------------------------------------------------------------------
def _try_load_rf_onnx(rf_joblib_path):
    """<model>.joblib yaninda <model>.onnx varsa ONNX Runtime session'i yukler.
    Yoksa/yuklenemezse None doner -> rf_predict_proba joblib'e duser."""
    onnx_path = os.path.splitext(rf_joblib_path)[0] + ".onnx"
    if not os.path.exists(onnx_path):
        return None
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        prob_name = sess.get_outputs()[1].name  # [label, probabilities]
        print(f"    [ONNX] RF hizlandirma aktif: {os.path.basename(onnx_path)}")
        return {"sess": sess, "in_name": in_name, "prob_name": prob_name}
    except Exception as e:
        print(f"    [ONNX] yuklenemedi ({e}); joblib RF'ye dusuluyor.")
        return None


def _try_load_hgb_onnx(hgb_joblib_path):
    """(R28/O2.1) MainHGB_v5.onnx yukleyici -- _try_load_rf_onnx ile ayni desen.
    Dosya yoksa/yuklenemezse None (konsey sessizce devre disi = fail-safe)."""
    onnx_path = os.path.splitext(hgb_joblib_path)[0] + ".onnx"
    if not os.path.exists(onnx_path):
        return None
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        print(f"    [ONNX] HGB (konsey) aktif: {os.path.basename(onnx_path)}")
        return {"sess": sess, "in_name": sess.get_inputs()[0].name,
                "prob_name": sess.get_outputs()[1].name}
    except Exception as e:
        print(f"    [ONNX] HGB yuklenemedi ({e}); konsey vetosu DEVRE DISI.")
        return None


def committee_dissents(cfg, scaled_input):
    """(R28/O2.1) RF VE HGB ikisi de BENIGN'e >= esik guvenle katiliyorsa True.
    process_pcap.py::committee_dissents ile BIREBIR ayni mantik/esik."""
    if not R28_COMMITTEE:
        return False
    ox = cfg.get("hgb_onnx")
    if ox is None:
        return False
    try:
        p_ben_rf = float(rf_predict_proba(cfg, scaled_input)[0][cfg["RF_BENIGN_POS"]])
        if p_ben_rf < COMMITTEE_VETO_THRESHOLD:
            return False
        X = np.ascontiguousarray(scaled_input, dtype=np.float32)
        out = ox["sess"].run([ox["prob_name"]], {ox["in_name"]: X})[0]
        p_ben_hgb = float(np.asarray(out, dtype=np.float64)[0][0])
    except Exception as e:
        print(f"    [konsey] ONNX calisma-zamani hatasi, veto atlaniyor: {e}")
        return False
    return p_ben_hgb >= COMMITTEE_VETO_THRESHOLD


def committee_waives_early_attack(cfg, scaled_input):
    """(R34) True = RF VE HGB ikisi de bu paketi BENIGN sayiyor -> R12
    early_attack MUAFIYETI VERILMEZ (olgunluk/E/F kapilari yurulukte kalir).

    Gerekce ve olcum icin bkz. EARLY_ATTACK_MIN_SUSP yorum blogu. Fail-safe
    yon R26'nin TERSIDIR: konsey yoksa/hata verirse False doner, yani
    muafiyet ESKI davranisla verilir (yeni kapi no-op olur)."""
    if not R28_COMMITTEE:
        return False
    ox = cfg.get("hgb_onnx")
    if ox is None:
        return False
    try:
        p_ben_rf = float(rf_predict_proba(cfg, scaled_input)[0][cfg["RF_BENIGN_POS"]])
        if p_ben_rf < COMMITTEE_WAIVER_BENIGN_P:
            return False          # RF katilmiyor -> HGB'yi hic calistirma (ucuz cikis)
        X = np.ascontiguousarray(scaled_input, dtype=np.float32)
        out = ox["sess"].run([ox["prob_name"]], {ox["in_name"]: X})[0]
        p_ben_hgb = float(np.asarray(out, dtype=np.float64)[0][0])
    except Exception as e:
        print(f"    [konsey/muafiyet] ONNX calisma-zamani hatasi, atlaniyor: {e}")
        return False
    return p_ben_hgb >= COMMITTEE_WAIVER_BENIGN_P


def rf_predict_proba(cfg, scaled_input):
    """RF.predict_proba esdegeri [n,2] olasilik matrisi.

    ONNX session varsa onu kullanir (~900x hizli), yoksa joblib RF'ye duser.
    ONNX ciktisi joblib ile float32 hassasiyetinde (max|d|~6.6e-7, ALERT_THRESHOLD
    karar degisimi=0) ESDEGER -> MCC/DR/FAR/R8-R13 kalibrasyonu korunur.

    NOT (skl2onnx TreeEnsembleClassifier kusuru): ONNX'in 0. olasilik sutunu
    yanlis taban-degerle uretiliyor (P0 yerine P0-1); 1. sutun (P-ATTACK) DOGRU.
    Bu yuzden matris dogru 1. sutundan yeniden insa edilir: P0 = 1 - P1."""
    ox = cfg.get("rf_onnx")
    if ox is None:
        return cfg["main_rf"].predict_proba(scaled_input)
    X = np.ascontiguousarray(scaled_input, dtype=np.float32)
    out = ox["sess"].run([ox["prob_name"]], {ox["in_name"]: X})[0]
    p1 = np.asarray(out, dtype=np.float64)[:, 1]
    return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# TEK PAKET ICIN DISPATCHER (offline_test.ipynb cell 12 mantigi)
# ---------------------------------------------------------------------------
def dispatch_packet(packet, flow_table, cfg):
    master_features, flow_key = flow_table.extract_universal_features(packet)
    if master_features is None:
        master_features = {col: 0.0 for col in cfg["RAW_COLUMNS"]}
        master_features["PSH Flag Count"] = 1.0
        flow_key = None

    rate_value = float(master_features.get(cfg["VOLUMETRIC_FEATURE"], 0.0))
    (is_flood, vol_pkts, vol_syn, vol_ratio, is_scan, distinct_ports,
     is_conn_flood, distinct_conns) = flow_table.update_volumetric(packet)

    # ── (R10) TEK-PAKETLIK-AKIS RATE YAPAYLIGI: rate sinyalini kapila ─────
    # KOK NEDEN: extract_universal_features satir 365'te
    #   flow_duration_sec = max((time_epoch - start_time)/1e6, 1e-6)
    # bir akisin ILK paketinde time_epoch == start_time -> duration=1e-6 ->
    #   Flow Packets/s = total_pkts / 1e-6 = 1/1e-6 = 1_000_000
    # Bu deger GERCEK hiz degil, "tek paketlik akisin sifir-sureli" yapaylik.
    # 1_000_000 > VOLUMETRIC_THRESHOLD (500_000) oldugundan, HER yeni akisin
    # ILK paketi rate-tetikleyici ile RF (volumetrik) dalina gidiyordu.
    # canli_trafik_kaydi.csv'de bu, 192.168.10.15'in cesitli CDN/HTTP host'larina
    # actigi 698 benign tek-paketlik akisi (dst_port=80'de 100 FP dahil) RF'ye
    # gonderip ATTACK skorlatiyordu (@0.5'te 105 FP'nin 100'u). Sticky is_scan
    # DEGIL (.15 hic is_scan=True olmuyor), is_flood DE degil -> kaynak: rate.
    # DUZELTME: rate-tetikleyici yalnizca akis >=2 pakete ulastiginda gecerli
    # (gercek hiz ancak ikinci paketin IAT'siyle olculur). is_flood/is_scan
    # ETKILENMEZ -> gercek SYN-flood/DoS-Hulk volumetrik dalda kalir.
    # KRITIK: DoS-Hulk (172.16.0.1<->192.168.10.50:80) 2039 paketi is_flood
    # ile, kalan rate-only paketleri ise LSTM+Counter dalinda (377/381 TP)
    # zaten yakalandigi icin TP=3067 KORUNUR (R10 oncesi/sonrasi dogrulandi).
    flow_total_pkts = 0
    if flow_key is not None:
        _f = flow_table.flows.get(flow_key)
        if _f is not None:
            flow_total_pkts = _f.get("fwd_packets", 0) + _f.get("bwd_packets", 0)
    # ── (R11) SIFIR-SURELI RATE YAPAYLIGI (clamp tabani): rate sinyalini kapila ─
    # KOK NEDEN: extract_universal_features satir 365'te
    #   flow_duration_sec = max((time_epoch - start_time)/1e6, 1e-6)
    # IKI (veya daha fazla) paket AYNI time_epoch cozunurlugunde gelince
    # (time_epoch - start_time) ~ 0 -> duration 1e-6 TABANINA clamp'lenir ->
    #   Flow Packets/s = total_pkts / 1e-6 = total_pkts * 1_000_000
    # R10 (flow_total_pkts>=2) bu durumu YAKALAYAMAZ cunku paket gercekten 2.
    # paket; ama olculmus GERCEK bir IAT yok -> rate astronomik ama YAPAY.
    # canli_trafik_kaydi.csv'de bu, 192.168.10.15:51230->54.192.36.157:80 saf-ACK
    # paketini RF'ye gonderip risk=0.61 ATTACK skorlatti (R11 Madde-2 FP).
    # DUZELTME: rate_value clamp tabanina yapismis ise (rate >= total_pkts*1e6*0.99,
    # yani duration <= ~1.01e-6) bu "olculmemis hiz" sayilir -> rate_trigger=False.
    # GUVENLI: bu imzayi tasiyan TUM 9 rate_trigger adayi BENIGN'dir (8'i DNS,
    # 1'i bu ACK); DoS-Hulk bu imzayi HIC tasimaz (gercek flood'da duration>1e-6
    # oldugundan rate < pkts*1e6) -> rate-only DoS bagimliligi=0, TP=3067 KORUNUR.
    # is_flood/is_scan ETKILENMEZ (deterministik SYN-burst/scan sayaclari).
    # (R11 Madde-2 RATE-CLAMP FIX KASITLI ATLANDI -- SLOW-ATTACK DR REGRESYONU)
    # Tasarlanan fix: rate_value clamp tabanina yapismissa (rate >= total_pkts*1e6*0.99,
    # duration<=~1.01e-6) rate_trigger=False. canli_trafik_kaydi.csv'de bu, tek RF
    # FP'sini (192.168.10.15:51230->54.192.36.157:80) RF dalindan cikariyordu.
    # ANCAK validate_slow_attacks.py ile olculen DR regresyonu yaratti:
    #   Slowloris    DR 0.9234 -> 0.6256 (RF paket 1363->912, TP 4765->3228)
    #   Slowhttptest DR 0.9775 -> 0.9424 (TP 12972->12506)
    # KOK NEDEN: Slowloris/Slowhttptest paketleri de cogu zaman ayni time_epoch
    # cozunurlugunde toplanip clamp imzasi tasir -> baseline'da rate_trigger ile
    # RF (volumetrik) dalinda yuksek skorla yakalaniyordu. Clamp-kapisi bunlari
    # LSTM dalina kaydirir -> LSTM zayif skorlar -> FN. YANI slow tespiti, R10'un
    # giderdigi tek-paket yapaylgina benzer sekilde, rate-clamp yapaylgina da
    # KAZARA bagimliymis. AYRICA: bu fix uygulansa bile canli_trafik FP'si NET
    # DUSMUYORDU (paket RF'den cikinca LSTM dalinda risk=1.0 ile yine FP kaliyor
    # -- 4 FP'nin DORDU DE ayni LSTM model davranisi). KARAR: net FP kazanci=0,
    # slow-DR kaybi gercek -> rate_trigger R10 halinde BIRAKILDI (clamp-kapisi yok).
    # ── (R33) SABIT-PAYDALI KAYAN PENCERE HIZI -- process_pcap.py ile parite ──
    # Gerekce ve tam olcum process_pcap.py'deki R33 yorumunda.
    if R33_RATE_WINDOW:
        _now_s = _to_float(packet.get("frame.time_epoch"), 0.0)
        rate_win_pkts = flow_table.update_rate_window(flow_key, _now_s) if flow_key is not None else 0
        rate_trigger = rate_win_pkts >= RATE_WIN_MIN_PKTS
    else:
        rate_win_pkts = 0
        rate_trigger = (rate_value > cfg["VOLUMETRIC_THRESHOLD"]) and (flow_total_pkts >= 2)
    is_volumetric = bool(is_flood or is_scan or rate_trigger or is_conn_flood)

    # ── (R9) DNS/UDP-53 ISTISNASI: VOLUMETRIK DALDAN ZORLA CIKAR ──────
    # KOK NEDEN: (src,dst) ikilisi GECMISTE port-tarama esigini gecip is_scan=True
    # KALICI hale gelince o ikilinin SONRAKI TUM paketleri -kendi protokol/portuna
    # bakilmaksizin- RF (volumetrik) dalina gidiyordu (karar (src,dst) GECMISINE
    # bakar, MEVCUT paketin kendisine degil). canli_trafik_kaydi.csv'de
    # 192.168.10.3 (DNS sunucu) -> 192.168.10.15 (istemci) yanit trafigi bu ikiliyi
    # is_scan=True yapti (yanitlarin update_volumetric'teki ham udp.dstport'u
    # istemcinin her sorguda DEGISEN ephemeral portu -> 40+ farkli port ->
    # is_scan; NOT: bu kopyadaki update_volumetric'te R5 SYN-only filtresi YOK,
    # ports.add(dst_port) kosulsuz). Sonuc: bu ciftin DNS paketleri RF'ye gidip
    # ATTACK skorlandi -> @0.5'te 217 DNS FP (204'u .3->.15:53).
    # DUZELTME: MEVCUT paket DNS ise (UDP, srcport==53 VEYA dstport==53,
    # TCP-flag YOK) is_volumetric=False zorla -> LSTM+Counter dala git.
    # DAR KAPSAM: gercek nmap SYN-scan / SYN-flood TCP'dir (udp portu yok, SYN
    # tasir) -> hic etkilenmez (CSV'de 2253 TCP-SYN paketi, hicbiri bu kosulda).
    _udp_sport = _to_int(packet.get("udp.srcport"), 0)
    _udp_dport = _to_int(packet.get("udp.dstport"), 0)
    _tcp_flags_int = _parse_tcp_flags(packet.get("tcp.flags"))
    is_dns_packet = (_udp_sport == 53 or _udp_dport == 53) and _tcp_flags_int == 0
    if is_dns_packet and is_volumetric:
        is_volumetric = False

    if is_volumetric:
        row_df = pd.DataFrame(
            [{col: float(master_features.get(col, 0.0)) for col in cfg["RAW_COLUMNS"]}],
            columns=cfg["RAW_COLUMNS"],
        )
        scaled_input = cfg["PIPELINE"].transform(row_df)
        rf_proba = rf_predict_proba(cfg, scaled_input)[0]   # ONNX (varsa) / joblib
        p_benign_rf = float(rf_proba[cfg["RF_BENIGN_POS"]])
        rf_risk = float(1.0 - p_benign_rf)

        # (R28/O1.1-O1.3) process_pcap.py ile parite -- gerekce/olcum orada.
        if is_flood:
            flood_conf = float(vol_ratio * min(vol_syn / (SYN_BURST_MIN * 3), 1.0))
            if R28_SYN_ABS and vol_syn >= SYN_ABS_MIN:
                flood_conf = max(flood_conf, float(min(vol_syn / SYN_ABS_FULL, 1.0)))
            risk_score = max(rf_risk, flood_conf)
        elif is_conn_flood:
            # (R28/O1.3) KAPI-TUTARLI RAMPA: kapinin ACILDIGI noktada guven
            # ZATEN esik-ustu olmali (O1.1'deki mutlak-SYN kolunun aynisi).
            # (conns-MIN)/(FULL-MIN) formulu kapida 0 veriyor -> paket RF dalina
            # tasinip imzasiz kaliyor ve FN uretiyordu (ilk denemede olculdu).
            conn_conf = float(min(distinct_conns / CONN_FLOOD_FULL, 1.0))
            risk_score = max(rf_risk, conn_conf)
        elif is_scan:
            scan_conf = float(min(
                max(distinct_ports - PORT_SCAN_MIN_PORTS, 0) /
                (PORT_SCAN_FULL_PORTS - PORT_SCAN_MIN_PORTS), 1.0))
            # (R28/O1.2) RF-veto: RF argmax BENIGN VE port cesitliligi benign
            # bandindaysa imza tabani iptal (risk tamamen RF'e kalir).
            if R28_SCAN_RF_VETO and int(np.argmax(rf_proba)) == cfg["RF_BENIGN_POS"] \
               and distinct_ports < SCAN_RF_VETO_MAX_PORTS:
                scan_conf = 0.0
            risk_score = max(rf_risk, scan_conf)
        else:
            risk_score = rf_risk
        model = "RF"
        counter_score = None
    else:
        row_df = pd.DataFrame(
            [{col: float(master_features.get(col, 0.0)) for col in cfg["RAW_COLUMNS"]}],
            columns=cfg["RAW_COLUMNS"],
        )
        scaled_input = cfg["PIPELINE"].transform(row_df)
        n_feat_expected = cfg["n_feat_lstm"]
        curr_feat = scaled_input.shape[1]
        if curr_feat < n_feat_expected:
            pad = np.zeros((scaled_input.shape[0], n_feat_expected - curr_feat))
            lstm_in_data = np.hstack([scaled_input, pad])
        else:
            lstm_in_data = scaled_input[:, :n_feat_expected]
        lstm_input = lstm_in_data.reshape(1, 1, n_feat_expected).astype("float32")
        # PERF: derlenmis tf.function (load_models'te trace edildi) -> birebir ayni
        # forward pass, ama per-call overhead'siz.
        raw_pred = np.asarray(cfg["lstm_fast"](lstm_input))[0]
        if len(raw_pred) == 1:
            p_attack = float(raw_pred[0])
            proba = np.array([1.0 - p_attack, p_attack])
        else:
            proba = raw_pred
        lstm_score = float(1.0 - proba[cfg["BENIGN_INDEX"]])

        if flow_key is not None and flow_key in flow_table.flows:
            counter_score, susp_count, prog_pkts, _ = flow_table.update_progressive(packet, flow_table.flows[flow_key])
        else:
            counter_score, susp_count, prog_pkts = 0.0, 0, 0
        blended = LSTM_WEIGHT * lstm_score + COUNTER_WEIGHT * counter_score
        risk_score = float(max(lstm_score, counter_score, blended))

        # ── (R12) AKIŞ-OLGUNLUĞU KAPISI -- YENİDEN TASARLANDI ────────────────
        # (R8 SORUNU) Eski kapı: immature akışta (prog_pkts<PROG_MIN_PKTS, Counter
        # henüz konuşmamış) yalnızca lstm_score>=LSTM_ALERT_FLOOR(0.95) ALERT
        # üretebiliyordu; aksi halde risk_score eşik-altına (THR-1e-3=0.399)
        # SABİTLENİYORDU. DISPATCH_ALERT_THRESHOLD==ALERT_THRESHOLD==0.4 olduğundan
        # cap HER ZAMAN eşiğin az altındaydı -> immature akışların ilk 11 paketi
        # lstm>=0.95 olmadıkça ASLA alarm üretemiyordu.
        # (ÖLÇÜLEN REGRESYON) validate_slow_attacks.py: bu kapı pilota R11'de port
        # edilince Slowloris DR 0.9234->0.6256, Slowhttptest 0.9775->0.9424.
        # (R12 KÖK NEDEN -- _scratch_r12_flip.py kanıtı) Kapının bastırdığı 200
        # GERÇEK Slowloris saldırı paketinin lstm_score'u 0.85-0.94 bandında
        # (med=0.916, min=0.508) -> 0.95 FLOOR'un HEMEN ALTINDA; risk LSTM'den
        # geliyor (precap==lstm, Counter katkısı 0). R8'in asıl hedefi benign
        # CDN/HTTP burst'leri ise lstm 0.6-0.92 bandında AMA susp_count=0.
        # (R12 AYIRT EDİCİ -- _scratch_r12_2d.py kanıtı) Bastırılan immature
        # paketlerde (susp_count>=1 & lstm>=0.80) hücresi: Slowloris ATTACK=144 vs
        # BENIGN=3 (48:1). Benign burst'ler susp_count=0 hücrelerinde toplanır.
        # YANİ saldırı imzasını (penceede >=1 suspicious paket: is_syn_only /
        # yarı-açık bağlantı) lstm yüksek bandıyla birleştirmek iki sınıfı ayırır.
        # (R12 ÇÖZÜM) "early_attack" istisnası: immature akış pencerede >=1
        # suspicious paket TAŞIYOR ve lstm_score>=EARLY_ATTACK_LSTM ise kapı
        # ATLANIR (bastırılmaz). Aksi halde (kanıtsız immature: susp_count=0 VEYA
        # lstm düşük) eski R8 davranışı korunur -> benign burst bastırılır.
        # (ÖLÇÜM -- _scratch_r12_candidate.py, 2-batch+GAP) lstm>=0.85 & cnt>=1:
        #   Slowloris DR 0.8344->0.9238 (~BEFORE_R11 0.9234), Slowhttptest
        #   0.9984->0.9987, GAP_clean FAR +0.0047 (tam doğrulama ADIM 3'te).
        _flow_now = flow_table.flows.get(flow_key) if flow_key is not None else None
        immature_flow = prog_pkts < PROG_MIN_PKTS and counter_score < DISPATCH_ALERT_THRESHOLD
        early_attack = (susp_count >= EARLY_ATTACK_MIN_SUSP) and (lstm_score >= EARLY_ATTACK_LSTM)
        # (R34) Konsey guvencesi -- yalnizca muafiyet ADAYI paketlerde cagrilir
        # (maliyet amortize: bu korpusta paketlerin ~%4'u).
        if early_attack and committee_waives_early_attack(cfg, scaled_input):
            early_attack = False
        # (R13) KURULU-ISTEMCI INIT-WINDOW ISTISNASI (Madde-1 FP fix), pp<=1 ile
        # DARALTILMIS HALI.
        # init_win_fwd kucuk (<=512) => bu paket, capture'in BASLADIGI noktada
        # SYN'i gorulmemis bir akisin "ilk-gorulen" (post-handshake kucuk window)
        # paketidir. canli_trafik_kaydi.csv'deki 3 FP (192.168.10.15->CDN:80,
        # pp=1, lstm=0.986-0.994, iwf=251-258, susp=0) bu desene uyar.
        # (REGRESYON -- pp KOSULSUZ ILK DENEME, REDDEDILDI) _scratch_madde1_slowcheck.py:
        # pp sarti OLMADAN eklendiginde, AYNI "SYN gorulmemis" artefakti saldiri
        # akislarinin pp=2 paketlerinde de olusuyor (iwf=229): Slowhttptest 258 +
        # Slowloris 2 = 260 GERCEK saldiri paketi yanlislikla bastirildi (DR
        # 0.9496->0.9301 / 0.7401->0.7391 -- R12 kazancinin cogu geri gitti).
        # pp=1 saldiri paketlerinde iwf HER ZAMAN 29200 (gercek SYN gorulmus);
        # artefakt SADECE pp>=2'de saldiri akislarinda da ortaya cikabiliyor.
        # (R13 NIHAI -- prog_pkts<=1 sarti) client_established SADECE flow_table'in
        # "ilk kez gordugu" pakette (pp<=1) degerlendirilir -> 3 FP (pp=1) yakalanir,
        # 260 slow-attack flip (pp=2) kapsam disi kalir. Bu sinyal, lstm tek-paket
        # asiri-guvenini (lstm>=FLOOR) bile gecersiz kilar. Mevcut R8 kosulu
        # (lstm<FLOOR) OR ile korunur. DOGRULAMA: offline_validation_summary.json
        # (FP 4->1, MCC 0.9981->0.9988, DoS-Hulk TP=3067/FN=4 korunur) ve
        # slow_attack_validation_summary.json (Slowloris/Slowhttptest DR R12
        # seviyesine -- 0.7401/0.9496 -- geri doner).
        init_win_fwd = float(master_features.get("Init_Win_bytes_forward", 0.0))
        # (R17/R18 UYUMU) R13'un "pp<=1" sarti "flow_table'in ILK GORDUGU paket"
        # anlamina geliyordu. R17 flip'inden sonra ayni artefakt (SYN'i gorulmemis
        # akisin ilk ISTEMCI paketi, post-handshake kucuk window) pp>1'de olusur --
        # cunku sunucu yonundeki paketler zaten sayilmistir. Flip EDILMIS akislarda
        # esdeger kosul "ilk forward paket"tir (fwd_packets<=1). Flip edilmemis
        # akislarda davranis AYNEN korunur (260 slow-attack paketini geri getiren
        # kosulsuz-pp denemesi hala reddedilmis durumda).
        _fp_now = _flow_now.get("fwd_packets", 0) if _flow_now is not None else 0
        _first_client_pkt = (prog_pkts <= 1) or (
            R17_ORIENT and _flow_now is not None and _flow_now.get("flipped") and _fp_now <= 1
        )
        client_established = _first_client_pkt and init_win_fwd <= INIT_WIN_CLIENT_MAX
        # (R28/O3.1) surekli olgunluk cezasi OLCULDU ve REDDEDILDI -- gerekce
        # ve olcum process_pcap.py'deki ayni-adli yorum blogunda.
        if immature_flow and (not early_attack) and \
           (lstm_score < LSTM_ALERT_FLOOR or client_established):
            risk_score = min(risk_score, DISPATCH_ALERT_THRESHOLD - 1e-3)

        # ── (R18) YONU DOGRULANMAMIS AKIS KAPISI (R17 orientation'in yan etkisi) ──
        # R17 mid-stream yakalanan bir akista "forward = istemci" yonunu SYN
        # bayragi/efemeral-port sezgiseliyle FLIP edebilir. Flip edildigi anda
        # akisin forward tarafindan HENUZ HIC PAKET GORULMEMISTIR:
        #   fwd_lengths=[] -> Total Length of Fwd Packets/Fwd Packet Length Max/
        #   Mean = 0, Fwd/Bwd Byte Ratio = 0, Init_Win_bytes_forward = mevcut
        #   (BACKWARD!) paketin window'una duser, Down/Up Ratio = bwd/1.
        # Yani oznitelik vektorunun forward yarisi UYDURMADIR (egitimde boyle bir
        # sey yok) -> LSTM bu OOD vektore 0.96-1.00 skorlar. OLCULDU
        # (validate_offline_predictions --models v5, canli_trafik dilimi):
        # R17-oncesi FP=1 -> yalniz-orientation FP=17.
        # KAPI: forward tarafindan en az 1 paket gorulene kadar skor esik-altina
        # sabitlenir. DAR KAPSAM: fwd_packets==0 SADECE flip edilmis akista
        # olusabilir (flip yoksa ilk paket taniminda forward'dir) -> SYN ile
        # baslayan tum akislar (CICIDS'te ~%99) ETKILENMEZ.
        if R18_ORIENT_GATE and _flow_now is not None and _flow_now.get("fwd_packets", 0) == 0:
            risk_score = min(risk_score, DISPATCH_ALERT_THRESHOLD - 1e-3)

        # (R28/O2.1) davranissal-dal konsey vetosu -- process_pcap.py parite.
        # Yalnizca hala esik-ustu paketlerde cagrilir (ONNX maliyeti amortize).
        if risk_score >= DISPATCH_ALERT_THRESHOLD and committee_dissents(cfg, scaled_input):
            risk_score = min(risk_score, DISPATCH_ALERT_THRESHOLD - 1e-3)
        model = "LSTM+Counter"

    return {
        "src_ip": packet.get("ip.src", "0.0.0.0"),
        "dst_ip": packet.get("ip.dst", "0.0.0.0"),
        "dst_port": master_features.get("Destination Port"),
        "is_volumetric": is_volumetric,
        "model": model,
        "risk_score": risk_score,
    }


# ---------------------------------------------------------------------------
# GROUND TRUTH ETIKETLEME (5-tuple, pcap-bazli)
# ---------------------------------------------------------------------------
def gt_label(packet):
    """1 = ATTACK (CICIDS-2017 Wednesday DoS), 0 = BENIGN."""
    s, d = packet.get("ip.src", ""), packet.get("ip.dst", "")
    sp = _to_int(packet.get("tcp.srcport"), 0)
    dp = _to_int(packet.get("tcp.dstport"), 0)
    pair = {s, d}
    if pair == {ATTACKER_IP, VICTIM_IP} and (dp == VICTIM_PORT or sp == VICTIM_PORT):
        return 1
    return 0


# ---------------------------------------------------------------------------
# PCAP -> paket sozlukleri (tshark)
# ---------------------------------------------------------------------------
def read_pcap(pcap_path):
    cmd = [TSHARK, "-r", pcap_path, "-T", "fields"]
    for fld in TSHARK_FIELDS:
        cmd += ["-e", fld]
    cmd += ["-E", "separator=,", "-E", "header=y", "-E", "occurrence=f"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tshark failed ({proc.returncode}) on {pcap_path}: {proc.stderr[:300]}")
    reader = csv.DictReader(io_lines(proc.stdout))
    return list(reader)


def io_lines(text):
    import io as _io
    return _io.StringIO(text)


def find_batch(idx):
    pat = os.path.join(PCAP_DIR, f"Wednesday-workingHours_{idx}_*.pcap")
    matches = glob.glob(pat)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# METRIK YARDIMCILARI
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def metrics_from_conf(tp, fp, tn, fn):
    import math
    dr = tp / (tp + fn) if (tp + fn) else 0.0          # Detection Rate = Recall(attack)
    far = fp / (fp + tn) if (fp + tn) else 0.0          # False Alarm Rate
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1_a = 2 * prec * dr / (prec + dr) if (prec + dr) else 0.0
    # benign F1
    prec_b = tn / (tn + fn) if (tn + fn) else 0.0
    rec_b = tn / (tn + fp) if (tn + fp) else 0.0
    f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) else 0.0
    macro_f1 = (f1_a + f1_b) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return {
        "DR_recall_attack": round(dr, 4), "FAR": round(far, 4),
        "precision_attack": round(prec, 4), "F1_attack": round(f1_a, 4),
        "F1_benign": round(f1_b, 4), "Macro_F1": round(macro_f1, 4),
        "MCC": round(mcc, 4), "accuracy": round(acc, 4),
    }


# ---------------------------------------------------------------------------
# ANA
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(TSHARK):
        print(f"[HATA] tshark bulunamadi: {TSHARK}")
        sys.exit(1)

    cfg = load_models()
    selected = [(b, "attack") for b in ATTACK_BATCHES] + [(b, "clean") for b in CLEAN_BATCHES]

    all_rows = []
    batch_summ = []
    glob_true, glob_pred = [], []

    for idx, expected in selected:
        pcap = find_batch(idx)
        if not pcap:
            print(f"[!] batch {idx} bulunamadi, atlaniyor.")
            continue
        print(f"\n[batch {idx} | beklenen={expected}] {os.path.basename(pcap)}", flush=True)
        try:
            packets = read_pcap(pcap)
        except Exception as e:
            print(f"    [HATA] {e}", flush=True)
            continue
        print(f"    {len(packets)} paket okundu, dispatcher calisiyor...", flush=True)

        # her batch icin TAZE flow table (offline_test tek dosya isliyordu;
        # batch'ler bagimsiz -> state sizmasini onler, gercekci stream segmenti)
        flow_table = UniversalRealTimeFlowTable()
        y_true, y_pred, risks = [], [], []
        n_rf = n_lstm = 0

        for pkt in packets:
            gt = gt_label(pkt)
            res = dispatch_packet(pkt, flow_table, cfg)
            alert = 1 if res["risk_score"] >= ALERT_THRESHOLD else 0
            y_true.append(gt)
            y_pred.append(alert)
            risks.append(res["risk_score"])
            if res["model"] == "RF":
                n_rf += 1
            else:
                n_lstm += 1
            all_rows.append({
                "batch": idx, "expected": expected,
                "src_ip": res["src_ip"], "dst_ip": res["dst_ip"],
                "dst_port": res["dst_port"], "model": res["model"],
                "is_volumetric": res["is_volumetric"],
                "risk_score": round(res["risk_score"], 4),
                "gt_attack": gt, "alert": alert,
            })

        glob_true += y_true
        glob_pred += y_pred
        tp, fp, tn, fn = confusion(y_true, y_pred)
        n_alert = sum(y_pred)
        n_attack = sum(y_true)
        risks_np = np.array(risks) if risks else np.array([0.0])
        bs = {
            "batch": idx, "expected": expected, "pcap": os.path.basename(pcap),
            "total_pkts": len(packets),
            "gt_attack_pkts": n_attack, "gt_benign_pkts": len(packets) - n_attack,
            "alerts": n_alert,
            "mean_risk": round(float(risks_np.mean()), 4),
            "max_risk": round(float(risks_np.max()), 4),
            "n_RF": n_rf, "n_LSTM": n_lstm,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        }
        bs.update(metrics_from_conf(tp, fp, tn, fn))
        batch_summ.append(bs)
        print(f"    pkts={len(packets)} gt_attack={n_attack} alerts={n_alert} "
              f"RF={n_rf} LSTM={n_lstm} | DR={bs['DR_recall_attack']} FAR={bs['FAR']} "
              f"meanRisk={bs['mean_risk']} maxRisk={bs['max_risk']}", flush=True)

    # GENEL (tum paketler birlestirilmis)
    gtp, gfp, gtn, gfn = confusion(glob_true, glob_pred)
    overall = {
        "total_pkts": len(glob_true),
        "gt_attack_pkts": sum(glob_true), "gt_benign_pkts": len(glob_true) - sum(glob_true),
        "alerts": sum(glob_pred),
        "TP": gtp, "FP": gfp, "TN": gtn, "FN": gfn,
    }
    overall.update(metrics_from_conf(gtp, gfp, gtn, gfn))

    # CIKTILAR
    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "batches": batch_summ}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("GENEL (paket-bazli, tum pilot batch'ler):")
    for k, v in overall.items():
        print(f"  {k:20s}: {v}")
    print(f"\n[+] Paket-bazli sonuc : {OUT_CSV}")
    print(f"[+] Ozet (JSON)       : {OUT_JSON}")


if __name__ == "__main__":
    main()
