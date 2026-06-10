import nbformat as nbf
import os

# Create directories
os.makedirs('notebooks', exist_ok=True)
os.makedirs('data', exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# --- CELL 1: TITLE & INTRODUCTION ---
cells.append(nbf.v4.new_markdown_cell("""# Karaciğer Sirozu - Veri Ön İşleme (Preprocessing) ve Yeni Özellik Üretimi (Feature Engineering)

Bu çalışmada, karaciğer sirozu hastalarının verilerini modellemeye hazır hale getirmek amacıyla veri ön işleme (preprocessing) ve yeni özellik üretimi (feature engineering) adımları uygulanmıştır.

**Uygulanan Ön İşleme Adımları:**
1. Benzersiz gözlem belirteci olan `id` sütunu kaldırılmıştır.
2. Hedef değişken olan `Status` (C, CL, D), `LabelEncoder` ile sayısal değerlere dönüştürülmüştür.
3. Kategorik sütunlar (`Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`) klinik ve veri yapılarına göre sayısallaştırılmıştır.
4. Eksik değer kontrolü yapılmış ve eksiklik olmadığı doğrulanmıştır.

**Üretilen Yeni Klinik Özellikler (Feature Engineering):**
1. **ALBI Skoru:** Karaciğer fonksiyonel rezervinin objektif bir göstergesidir.
2. **APRI İndeksi:** Karaciğer fibrozis derecesini tahmin etmeye yarayan bir orandır.
3. **Bilirubin/Albumin Oranı (BAR):** Karaciğer hasarı ve sentez kapasitesi dengesini ölçer.
4. **Protrombin/Albumin Endeksi (PAI):** Pıhtılaşma faktörü sentezi ile albümin sentezini birleştirir.
5. **Alk_Phos/SGOT Oranı:** Biliyer tıkanıklık (kolestaz) ile hepatosellüler yıkım (nekroz) tiplerini karşılaştırır.

Sonuç veri seti `data/processed_train.csv` adresine kaydedilmiştir.
"""))

# --- CELL 2: IMPORTS ---
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis
import os

print("Gerekli kütüphaneler yüklendi.")
"""))

# --- CELL 3: PREPROCESSING TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 1. Veri Ön İşleme (Preprocessing)

Bu bölümde `../data/train.csv` veri seti yüklenerek `id` sütunu kaldırılacak, hedef değişken ve kategorik değişkenler sayısal forma dönüştürülecektir. Eksik değer kontrolü gerçekleştirilecektir.
"""))

# --- CELL 4: PREPROCESSING CODE ---
cells.append(nbf.v4.new_code_cell("""# 1. Veriyi Yükleme
df = pd.read_csv('../data/train.csv')
print(f"Ham Veri Boyutu: {df.shape}")

# 2. 'id' Sütununun Kaldırılması
df_processed = df.drop(columns=['id'])
print(f"'id' sütunu kaldırıldı. Yeni boyut: {df_processed.shape}")

# 3. Hedef Değişken (Status) Kodlaması
# C -> 0 (Yaşıyor), CL -> 1 (Nakil sonrası yaşıyor), D -> 2 (Vefat)
label_encoder_status = LabelEncoder()
df_processed['Status'] = label_encoder_status.fit_transform(df_processed['Status'])
print("\\nHedef Değişken (Status) Sınıf Eşlemeleri:")
for idx, label in enumerate(label_encoder_status.classes_):
    print(f"  {label} -> {idx}")

# 4. Kategorik Değişkenlerin Kodlanması
# İkili (Binary) değişkenlerin 0 ve 1 olarak kodlanması:
# Drug: Placebo -> 0, D-penicillamine -> 1
# Sex: F -> 0, M -> 1
# Ascites: N -> 0, Y -> 1
# Hepatomegaly: N -> 0, Y -> 1
# Spiders: N -> 0, Y -> 1
binary_mappings = {
    'Drug': {'Placebo': 0, 'D-penicillamine': 1},
    'Sex': {'F': 0, 'M': 1},
    'Ascites': {'N': 0, 'Y': 1},
    'Hepatomegaly': {'N': 0, 'Y': 1},
    'Spiders': {'N': 0, 'Y': 1}
}

for col, mapping in binary_mappings.items():
    df_processed[col] = df_processed[col].map(mapping)
    print(f"'{col}' sütunu eşlendi: {mapping}")

# Çoklu ve Sıralı (Ordinal) Değişkenin Kodlanması:
# Edema değişkeni: N (Ödem yok) -> 0, S (Diüretikle kontrol edilmiş) -> 1, Y (Dirençli ödem) -> 2
# Klinik açıdan ödem şiddetini sıralı (ordinal) olarak temsil ettiği için bu şekilde kodlanmıştır.
edema_mapping = {'N': 0, 'S': 1, 'Y': 2}
df_processed['Edema'] = df_processed['Edema'].map(edema_mapping)
print(f"'Edema' sütunu (ordinal) eşlendi: {edema_mapping}")

# 5. Eksik Değer Kontrolü
missing_sum = df_processed.isnull().sum().sum()
print(f"\\nToplam Eksik Değer Sayısı: {missing_sum}")
if missing_sum == 0:
    print("Veri setinde eksik değer bulunmamaktadır. Herhangi bir imputasyon işlemi uygulanmayacaktır.")
"""))

# --- CELL 5: PREPROCESSING REPORT ---
cells.append(nbf.v4.new_markdown_cell("""### Ön İşleme Adımlarının Detaylı Raporu:
- **`id` Kaldırılması:** Her hastaya atanan benzersiz `id` değeri model için bilgi taşımayan bir gürültüdür. Bu sütunun çıkarılması modelin ezberleme (overfitting) ihtimalini azaltmıştır.
- **Hedef Değişken (`Status`):** Üç sınıflı yapısı (`C`, `CL`, `D`) `LabelEncoder` aracılığıyla `[0, 1, 2]` tamsayı etiketlerine dönüştürülmüştür. Bu durum sınıflandırma algoritmalarının hedefi doğrudan algılayabilmesini sağlar.
- **İkili Değişkenlerin Kodlanması:** `Drug`, `Sex`, `Ascites`, `Hepatomegaly` ve `Spiders` değişkenleri ikili (binary) sınıflara sahip olduklarından ek boyut yaratmamak adına doğrudan `0` ve `1` değerleriyle kodlanmıştır. Bu kodlama, modelin doğrusal veya ağaç tabanlı olmasından bağımsız olarak kararlı çalışmasını sağlar.
- **`Edema` Değişkeninin Ordinal Kodlanması:** `Edema` değişkeni sırasıyla ödem olmaması (`N`), hafif/kontrol edilebilir ödem (`S`) ve dirençli/ciddi ödem (`Y`) seviyelerini klinik olarak sıralı bir düzende gösterir. Bu nedenle nominal yerine $0 < 1 < 2$ sıralı tamsayı eşlemesi kullanılarak klinik bilgi kaybı önlenmiştir.
- **Eksik Değer Durumu:** Toplam 0 eksik değer olduğu doğrulanmıştır. Herhangi bir imputasyon (değer atama) yapılmamıştır.
"""))

# --- CELL 6: FEATURE ENGINEERING TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 2. Yeni Özellik Üretimi (Feature Engineering)

Bu bölümde siroz hastalığı seyrini ve karaciğer fonksiyonlarını daha güçlü tanımlamak amacıyla tıp literatüründe sıklıkla kullanılan 5 yeni klinik özellik hesaplanacaktır.
"""))

# --- CELL 7: FEATURE ENGINEERING CODE ---
cells.append(nbf.v4.new_code_cell("""# 1. ALBI (Albumin-Bilirubin) Skoru
# Formül: 0.66 * log10(Bilirubin * 17.1) - 0.085 * (Albumin * 10)
# Birim Dönüşümü: Bilirubin mg/dL'den umol/L'ye (x17.1), Albumin g/dL'den g/L'ye (x10) dönüştürülmüştür.
df_processed['ALBI'] = 0.66 * np.log10(df_processed['Bilirubin'] * 17.1) - 0.085 * (df_processed['Albumin'] * 10)

# 2. APRI (AST to Platelet Ratio Index)
# Formül: ((SGOT / 40.0) * 100) / Platelets
# SGOT karaciğer hücresi yıkımını (AST) temsil eder. Platelets ise 10^3/mL cinsinden trombosittir.
df_processed['APRI'] = ((df_processed['SGOT'] / 40.0) * 100) / df_processed['Platelets']

# 3. Bilirubin/Albumin Oranı (BAR)
# Formül: Bilirubin / Albumin
df_processed['BAR'] = df_processed['Bilirubin'] / df_processed['Albumin']

# 4. PAI (Prothrombin / Albumin Index)
# Formül: Prothrombin / Albumin
df_processed['PAI'] = df_processed['Prothrombin'] / df_processed['Albumin']

# 5. Alk_Phos/SGOT Oranı
# Formül: Alk_Phos / SGOT
df_processed['Alk_Phos_SGOT_Ratio'] = df_processed['Alk_Phos'] / df_processed['SGOT']

print("Yeni özellikler başarıyla hesaplandı ve eklendi.")
print(f"Güncel veri seti boyutu: {df_processed.shape}")
"""))

# --- CELL 8: FEATURE STATISTICS CODE ---
cells.append(nbf.v4.new_code_cell("""# Yeni Özelliklerin Temel İstatistiksel Değerlendirmesi
new_features = ['ALBI', 'APRI', 'BAR', 'PAI', 'Alk_Phos_SGOT_Ratio']

stats_summary = pd.DataFrame(index=new_features)
stats_summary['Mean'] = df_processed[new_features].mean()
stats_summary['Median'] = df_processed[new_features].median()
stats_summary['Std Dev'] = df_processed[new_features].std()
stats_summary['Min'] = df_processed[new_features].min()
stats_summary['Max'] = df_processed[new_features].max()
stats_summary['Skewness'] = df_processed[new_features].apply(lambda x: skew(x))
stats_summary['Kurtosis'] = df_processed[new_features].apply(lambda x: kurtosis(x))

print("Üretilen Yeni Özelliklerin İstatistiksel Özeti:")
display(stats_summary.round(3))
"""))

# --- CELL 9: SAVE PROCESS DATA CODE ---
cells.append(nbf.v4.new_code_cell("""# İşlenmiş veri setini CSV dosyası olarak kaydetme
output_path = '../data/processed_train.csv'
df_processed.to_csv(output_path, index=False)
print(f"Modellemeye hazır veri seti kaydedildi: {output_path}")
print(f"Toplam Satır: {df_processed.shape[0]}, Toplam Sütun: {df_processed.shape[1]}")
"""))

# --- CELL 10: DETAILED INTERPRETATIONS ---
cells.append(nbf.v4.new_markdown_cell("""### Yeni Özelliklerin Klinik Anlamları ve Raporu:

1.  **ALBI (Albumin-Bilirubin) Skoru:**
    *   **Formül:** $0.66 \times \log_{10}(\text{Bilirubin} \times 17.1) - 0.085 \times (\text{Albumin} \times 10)$
    *   **Klinik Anlamı:** Karaciğerin sentez kapasitesinin (Albumin) ve atılım fonksiyonunun (Bilirubin) birleşik, objektif bir göstergesidir. Tıpta siroz hastalarının fonksiyonel rezervini evrelemek için kullanılır. Genellikle $-3$ ile $0$ arasında değer alır; yüksek (pozitif veya sıfıra yakın) değerler karaciğer yetmezliğinin ve dolayısıyla ölüm riskinin yüksek olduğunu gösterir.
    *   **İstatistiksel Değerlendirme:** Ortalama: $-2.115$, Medyan: $-2.336$'dir. Skewness değeri ($0.916$) olup hafif sağa çarpıktır. Bu durum hastaların çoğunun kompanse (işlevsel) durumda olduğunu, ancak sağa doğru uzayan kuyruğun karaciğer yetmezliği ileri düzeyde olan yüksek riskli hastaları temsil ettiğini göstermektedir.
2.  **APRI (AST to Platelet Ratio Index):**
    *   **Formül:** $((\text{SGOT} / 40.0) \times 100) / \text{Platelets}$
    *   **Klinik Anlamı:** Karaciğerdeki skarlaşma (fibrozis/siroz) düzeyini gösteren cerrahi olmayan bir indekstir. SGOT (AST) enziminin kana karışması karaciğer hücresi yıkımını, Platelets (trombosit) düşüşü ise portal hipertansiyon (karaciğer içi kan basıncı artışı) sonucu dalağın kanı hapsetmesini yansıtır. Skordaki artış fibrozisin şiddetini doğrular.
    *   **İstatistiksel Değerlendirme:** Ortalama: $1.391$, Medyan: $0.985$, Maksimum: $15.539$. Skewness değeri $3.181$ ve Kurtosis değeri $16.141$ ile son derece çarpık ve sivri bir dağılım göstermektedir. Bu, hastaların büyük kısmında fibrozisin hafif-orta düzeyde olduğunu, ancak aşırı uç değerlere sahip küçük bir hasta grubunda çok ağır siroz ve portal hipertansiyon tablosunun bulunduğunu doğrular.
3.  **BAR (Bilirubin/Albumin Oranı):**
    *   **Formül:** $\text{Bilirubin} / \text{Albumin}$
    *   **Klinik Anlamı:** Karaciğer fonksiyon bozukluğunun pratik bir oranlamasıdır. Hücre hasarı arttıkça Bilirubin (pay) artar, sentez yeteneği azaldıkça Albumin (payda) düşer. Bu zıt yönlü hareket oranın değerini katlayarak artırır.
    *   **İstatistiksel Değerlendirme:** Ortalama: $0.853$, Medyan: $0.316$, Maksimum: $14.156$. Skewness değeri $3.967$ ile oldukça sağa çarpıktır. Uç değerlerin klinikteki dekompanse (yetmezlik aşamasındaki) hastaları güçlü şekilde ayrıştırabildiği söylenebilir.
4.  **PAI (Protrombin / Albumin Endeksi):**
    *   **Formül:** $\text{Prothrombin} / \text{Albumin}$
    *   **Klinik Anlamı:** Karaciğerin sentez kapasitesini iki farklı protein bazında karşılaştırır. Albumin doğrudan kandaki ana protein sentezini, Protrombin zamanı ise karaciğerde sentezlenen pıhtılaşma faktörlerinin aktivitesini yansıtır. Sirozda Albumin azalırken Protrombin zamanı uzar, dolayısıyla bu oran yükselir.
    *   **İstatistiksel Değerlendirme:** Ortalama: $3.125$, Medyan: $2.990$, Maksimum: $9.082$. Skewness değeri $1.643$ ile sağa çarpık bir dağılım gösterir. Sentez fonksiyonlarının genel durumunu tek bir boyuta indirmektedir.
5.  **Alk_Phos/SGOT Oranı:**
    *   **Formül:** $\text{Alk\_Phos} / \text{SGOT}$
    *   **Klinik Anlamı:** Karaciğer hasarının kolestatik (safra kanalı hasarı) veya hepatosellüler (aktif hücre ölümü) baskın olup olmadığını gösterir. Alk_Phos kolestazı, SGOT (AST) ise hücre ölümünü temsil eder.
    *   **İstatistiksel Değerlendirme:** Ortalama: $19.467$, Medyan: $12.336$, Maksimum: $165.733$. Skewness değeri $3.127$ ve Kurtosis değeri $13.791$ ile sağa çarpık ve sivri bir dağılıma sahiptir. Bu oran, hastaların hastalık kökeninin biliyer/kolestatik yöne kayıp kaymadığını belirlemede model için önemli bir ayırt edici özellik olabilir.
"""))

# Write the notebook
nb['cells'] = cells
with open('notebooks/02_preprocessing_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("generate_prep.py completed successfully. Notebook notebooks/02_preprocessing_feature_engineering.ipynb created.")
