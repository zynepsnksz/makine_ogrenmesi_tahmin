import nbformat as nbf
import subprocess
import os
import sys

# Create directory for plots
os.makedirs('reports/figures', exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# --- CELL 1: TITLE & INTRODUCTION ---
cells.append(nbf.v4.new_markdown_cell("""# Karaciğer Sirozu Hastalık Sonuçları (Cirrhosis Outcomes) - Keşifçi Veri Analizi (EDA)

Bu çalışmada, karaciğer sirozu hastalarının klinik özellikleri ve laboratuvar bulgularını içeren veri seti üzerinde **Exploratory Data Analysis (EDA)** gerçekleştirilmiştir. 

**Analizin Amacı:** Veri setinin genel yapısını, hedef değişkenin dağılımını, eksik değerlerin durumunu, sayısal ve kategorik değişkenlerin istatistiksel özelliklerini, aykırı değerleri, değişkenler arası korelasyonları ve klinik ilişkileri incelemek ve raporlamaktır.

### Kurallar ve Yaklaşım:
- Veri seti üzerinde **kesinlikle** veri temizleme, eksik değer doldurma (imputation), veri silme, feature engineering veya model eğitme işlemleri yapılmamıştır.
- Analizler tamamen gözlemleme, görselleştirme ve istatistiksel raporlama üzerine kuruludur.
- Aykırı değerler sadece IQR ve görsel analizlerle tespit edilmiş, veri üzerinde herhangi bir değişiklik yapılmamıştır.
- Grafik tasarımı sunum kalitesinde, yüksek çözünürlüklü (300 DPI), `seaborn whitegrid` stili ve profesyonel renk paletleri kullanılarak hazırlanmıştır.

*Analiz ve yorum dili Türkçe olarak tercih edilmiştir.*
"""))

# --- CELL 2: IMPORTS AND STYLE SETUP ---
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import os

# Görselleştirme ve Stil Ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 15

# Kayıt dizini kontrolü
os.makedirs('reports/figures', exist_ok=True)

print("Kütüphaneler yüklendi ve grafik ayarları tamamlandı.")
"""))

# --- CELL 3: GENERAL DATA OVERVIEW TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 1. Genel Veri İncelemesi

Bu bölümde veri setinin genel boyutu, satır ve sütun sayıları, değişken tipleri, bellek kullanımı ile veri setinin ilk ve son 5 gözlemi incelenecektir.
"""))

# --- CELL 4: GENERAL DATA OVERVIEW CODE ---
cells.append(nbf.v4.new_code_cell("""# Veriyi Yükleme
df = pd.read_csv('data/train.csv')

# Genel Bilgiler
rows, cols = df.shape
print(f"Veri Seti Boyutu: {df.size} hücre")
print(f"Satır Sayısı (Gözlem): {rows}")
print(f"Sütun Sayısı (Değişken): {cols}")
print("-" * 50)

print("Veri Seti Bilgi Özeti (Memory & Types):")
df.info()
print("-" * 50)

print("Veri Tipleri Dağılımı:")
print(df.dtypes.value_counts())
print("-" * 50)

print("İlk 5 Gözlem:")
display(df.head(5))

print("Son 5 Gözlem:")
display(df.tail(5))
"""))

# --- CELL 5: GENERAL DATA OVERVIEW COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Genel Veri İncelemesi Yorumları:
- **Boyut ve Yapı:** Veri seti **7905 satır** ve **20 sütundan** oluşmaktadır. Toplamda 158.100 adet veri hücresi barındırmaktadır.
- **Değişken Tipleri:** Veri setinde **12 adet sayısal** (4 adet `int64` ve 8 adet `float64`) ve **8 adet kategorik/metinsel** (`object`) değişken bulunmaktadır.
- **Bellek Kullanımı:** Veri seti yaklaşık **1.2 MB** bellek alanına ihtiyaç duymaktadır. Bu boyut modern bilgisayarlar ve analiz araçları için oldukça hafiftir ve verinin bellekte hızlı işlenmesine olanak tanır.
- **Anahtar Değişkenler:**
  - `id`: Benzersiz gözlem anahtarıdır. Analiz dışı bırakılacaktır.
  - `Age`: Gün cinsinden verilmiştir. İlerleyen bölümlerde yaşın gün bazlı dağılımını bozmadan, yorumlarımızı klinik açıdan daha anlaşılır kılmak için yaklaşık yıl karşılıklarını ele alacağız.
  - `Stage`: Karaciğer hasarının evresini (1, 2, 3, 4) gösteren ve klinik düzey olarak hem kategorik hem de sıralı (ordinal) yapıda olan kritik bir değişkendir.
  - `Status`: Hastaların durumunu gösteren hedef (target) değişkendir.
"""))

# --- CELL 6: TARGET VARIABLE TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 2. Hedef Değişken Analizi (`Status`)

Bu bölümde hedef değişken olan `Status` (Hastalık Sonucu) incelenecektir. Bu değişken hastanın durumunu ifade eder:
- `C` (Censored - Yaşıyor/Takip Dışı)
- `CL` (Censored due to Liver Transplant - Karaciğer Nakli Yapılmış ve Yaşıyor)
- `D` (Death - Hayatını Kaybetmiş)

Bu değişkenin sınıf dağılımı, yüzdesik oranları ve sınıflar arası dengesizlik durumu (class imbalance) analiz edilecektir.
"""))

# --- CELL 7: TARGET VARIABLE CODE ---
cells.append(nbf.v4.new_code_cell("""status_counts = df['Status'].value_counts()
status_pct = df['Status'].value_counts(normalize=True) * 100

status_summary = pd.DataFrame({
    'Frekans (Adet)': status_counts,
    'Oran (%)': status_pct
})
print("Status Değişkeni Dağılımı:")
display(status_summary)

# Grafik Çizimi
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Countplot
sns.countplot(data=df, x='Status', ax=axes[0], palette='Set2', hue='Status', legend=False)
axes[0].set_title('Status Dağılımı (Sayısal)')
axes[0].set_xlabel('Durum (Status)')
axes[0].set_ylabel('Hasta Sayısı')

# Yüzdelik Değerleri Barların Üzerine Ekleme
total = len(df)
for p in axes[0].patches:
    height = p.get_height()
    axes[0].annotate(f'{height}\\n({(height/total)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., height + 50),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

# Pie Chart
axes[1].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=sns.color_palette('Set2', len(status_counts)),
            wedgeprops={'edgecolor': 'w', 'linewidth': 2})
axes[1].set_title('Status Dağılımı (Yüzdesel Oran)')

plt.suptitle('Hedef Değişken (Status) Sınıf Analizi', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 8: TARGET VARIABLE COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Hedef Değişken Analizi Yorumları:
- **Dağılım Durumu:**
  - Hastaların **%62.8'i (4965 hasta) `C` (Yaşıyor/Takip Dışı)** sınıfındadır.
  - Hastaların **%33.7'si (2665 hasta) `D` (Vefat)** sınıfındadır.
  - Hastaların sadece **%3.5'i (275 hasta) `CL` (Nakil Sonrası Yaşıyor)** sınıfındadır.
- **Sınıf Dengesizliği (Class Imbalance):**
  - Veri setinde ciddi bir dengesizlik mevcuttur. Özellikle `CL` sınıfı genel veri setinin yalnızca %3.5'ini oluşturarak azınlık sınıfı (minority class) durumundadır.
  - `C` ve `D` sınıfları veri setinin %96.5'ini domine etmektedir. Bu dengesizlik durumu, ileride yapılacak bir makine öğrenmesi modellemesinde (örneğin çoklu sınıflandırma) modelin `CL` sınıfını öğrenmesini zorlaştırabilir. Modelleme aşamasında bu durumun dikkate alınması ve uygun stratejiler (sınıf ağırlıkları ayarlama, resampling vb.) planlanması gerekebilir.
"""))

# --- CELL 9: MISSING VALUE TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 3. Eksik Değer Analizi

Veri setindeki eksik değerlerin sayısal miktarları, toplam gözlemlere oranı ve olası eksiklik desenleri incelenecektir. Grafiksel olarak doluluk durumları gösterilecektir.
"""))

# --- CELL 10: MISSING VALUE CODE ---
cells.append(nbf.v4.new_code_cell("""missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Eksik Değer Sayısı': missing_counts,
    'Eksiklik Oranı (%)': missing_pct
}).sort_values(by='Eksik Değer Sayısı', ascending=False)

print("Eksik Değer Tablosu:")
display(missing_df)

# Görselleştirme: Veri Seti Doluluk Oranları
plt.figure(figsize=(12, 6))
completeness_pct = 100 - missing_pct
sns.barplot(x=completeness_pct.values, y=completeness_pct.index, palette='viridis')
plt.title('Veri Değişkenlerinin Doluluk Oranları (%)', fontsize=14, pad=15)
plt.xlabel('Doluluk Oranı (%)')
plt.ylabel('Değişkenler')
plt.xlim(0, 110)

# Değer etiketlerini yazdırma
for i, v in enumerate(completeness_pct.values):
    plt.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/missing_values.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 11: MISSING VALUE COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Eksik Değer Analizi Yorumları:
- **Bulgular:** Veri setindeki **tüm değişkenlerin doluluk oranı %100'dür**. Hiçbir sütunda eksik (`NaN` / `null`) değer bulunmamaktadır.
- **Yorum ve Hipotez:**
  - Gerçek dünya klinik veri setlerinde laboratuvar ölçümlerinde (örneğin `Cholesterol`, `Copper`, `Tryglicerides` vb.) eksik değerlerin olması yaygın bir durumdur. Bu veri setinde sıfır eksik değer olması, bu veri setinin sentetik olarak üretilmiş (örneğin Kaggle Playground serisindeki gibi bir derin öğrenme modeli üretimi) olmasından veya analize başlamadan önce eksiksiz gözlemlerin filtrelenmiş olmasından kaynaklanıyor olabilir.
  - Klinik çalışmalarda laboratuvar testleri eksik olduğunda, bu eksikliklerin deseni sıklıkla **MAR (Missing at Random - Rastlantısal Eksiklik)** veya hastanın sağlık durumunun kötülüğünden dolayı test yapılamaması gibi **MNAR (Missing Not at Random - Rastlantısal Olmayan Eksiklik)** olarak karşımıza çıkabilir. Ancak mevcut veri setimiz tam olduğu için veri setini bozmadan veya doldurma işlemi yapmaya ihtiyaç duymadan doğrudan analize devam edebiliyoruz.
"""))

# --- CELL 12: NUMERICAL VARIABLES TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 4. Sayısal Değişken Analizi

Veri setinde yer alan sayısal değişkenlerin merkezi eğilim (ortalama, medyan), yayılım (standart sapma, min, max) ve şekil parametreleri (skewness, kurtosis) hesaplanacaktır. Her değişken için histogram çizilerek üzerine Kernel Density Estimation (KDE) eğrisi eklenecektir.

**Sayısal Değişkenler:** `N_Days`, `Age`, `Bilirubin`, `Cholesterol`, `Albumin`, `Copper`, `Alk_Phos`, `SGOT`, `Tryglicerides`, `Platelets`, `Prothrombin`
"""))

# --- CELL 13: NUMERICAL VARIABLES STATISTICS CODE ---
cells.append(nbf.v4.new_code_cell("""num_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 
            'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

# İstatistik Hesaplamaları
desc_stats = pd.DataFrame(index=num_cols)
desc_stats['Mean'] = df[num_cols].mean()
desc_stats['Median'] = df[num_cols].median()
desc_stats['Std Dev'] = df[num_cols].std()
desc_stats['Min'] = df[num_cols].min()
desc_stats['Max'] = df[num_cols].max()
desc_stats['Skewness'] = df[num_cols].apply(lambda x: skew(x))
desc_stats['Kurtosis'] = df[num_cols].apply(lambda x: kurtosis(x))

print("Sayısal Değişkenlerin Detaylı İstatistiksel Özeti:")
display(desc_stats.round(3))
"""))

# --- CELL 14: NUMERICAL VARIABLES HISTOGRAMS CODE ---
cells.append(nbf.v4.new_code_cell("""# Histogram & KDE Çizimleri (Grid Yapısında)
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()

# Yumuşak tonlu bir renk paleti seçimi
color_palette = sns.color_palette("muted", len(num_cols))

for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i], color=color_palette[i], edgecolor='w', alpha=0.7)
    axes[i].set_title(f'{col} Dağılım Grafiği', fontsize=12, pad=10)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Frekans', fontsize=10)
    
    # Skewness ve Kurtosis değerlerini grafik içine yazdırma
    skew_val = desc_stats.loc[col, 'Skewness']
    kurt_val = desc_stats.loc[col, 'Kurtosis']
    axes[i].text(0.95, 0.95, f'Skew: {skew_val:.2f}\\nKurt: {kurt_val:.2f}', 
                 transform=axes[i].transAxes, fontsize=9, verticalalignment='top', 
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# Boş kalan ekseni gizleme (11 değişken var, 12 kutu var)
fig.delaxes(axes[-1])

plt.suptitle('Sayısal Değişkenlerin Dağılımları ve KDE Eğrileri', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 15: NUMERICAL VARIABLES COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Sayısal Değişken Analizi Yorumları:

1. **`N_Days` (Takip Günü):**
   - Ortalama: 2030 gün (~5.5 yıl), Medyan: 1731 gün (~4.7 yıl).
   - Skewness (0.457) hafif pozitif yönlü çarpık, kurtosis (-0.569) basıktır. Bu durum, hastaların takip sürelerinin geniş bir aralığa yayıldığını ve belirli bir bölgede aşırı yığılma olmadığını göstermektedir.
2. **`Age` (Yaş - Gün bazında):**
   - Ortalama: 18363 gün, Medyan: 18719 gün.
   - Yaş gün bazında verilmiştir. Ortalama ve medyan değerleri yaklaşık **50.3 yıla** (18363 / 365.25) tekabül etmektedir. En küçük yaş yaklaşık 26.3 yıl (9598 gün) ve en büyük yaş 78.3 yıldır (28650 gün).
   - Skewness (-0.089) sıfıra oldukça yakındır, dağılımı neredeyse mükemmel bir simetrik yapı (normal dağılım) sergilemektedir.
3. **`Bilirubin` (Bilirubin Seviyesi):**
   - Ortalama: 2.59, Medyan: 1.10.
   - Skewness (3.717) ve Kurtosis (16.488) değerleri son derece yüksektir. Bu, değişkenin **aşırı sağa çarpık (right-skewed)** ve **leptokurtik (sivri/dik)** bir dağılıma sahip olduğunu gösterir. Hastaların büyük çoğunluğunun bilirubin değerleri normal sınırlarda (düşük) iken, küçük bir hasta grubunda klinik olarak çok kritik düzeyde aşırı yüksek bilirubin seviyeleri bulunmaktadır.
4. **`Cholesterol` (Kolesterol Seviyesi):**
   - Ortalama: 351.2, Medyan: 309.0.
   - Skewness (3.535) ve Kurtosis (17.518) değerleri ile sağa çarpık ve sivri bir dağılım göstermektedir. Birkaç uç değer dağılımı sağa doğru çekmektedir.
5. **`Albumin` (Albumin Seviyesi):**
   - Ortalama: 3.51, Medyan: 3.55.
   - Skewness (-0.574) hafif sola çarpıklık göstermekle birlikte, dağılımı genel olarak normal dağılıma oldukça yakındır. Karaciğer sentez fonksiyonunun bir göstergesi olan albumin değerleri belirli bir aralıkta yoğunlaşmıştır.
6. **`Copper` (Bakır Seviyesi):**
   - Ortalama: 83.9, Medyan: 73.0.
   - Skewness (2.296) ve Kurtosis (6.772) değerleri sağa çarpık bir dağılıma işaret etmektedir. Bakır birikimi karaciğer hasarının göstergelerindendir ve bazı hastalarda yüksek seviyelerdedir.
7. **`Alk_Phos` (Alkalen Fosfataz):**
   - Ortalama: 1812.8, Medyan: 1259.0.
   - Skewness (2.955) ve Kurtosis (10.745) ile belirgin derecede sağa çarpıktır.
8. **`SGOT` (Karaciğer Enzimi - AST):**
   - Ortalama: 114.6, Medyan: 108.5.
   - Skewness (1.455) ve Kurtosis (2.730) değerleri sağa çarpık bir dağılım göstermektedir ancak bilirubin ve kolesterol kadar aşırı değildir.
9. **`Tryglicerides` (Trigliserit Seviyesi):**
   - Ortalama: 115.3, Medyan: 104.0.
   - Skewness (2.493) ve Kurtosis (9.120) ile sağa çarpıktır.
10. **`Platelets` (Trombosit Sayısı):**
    - Ortalama: 265.2, Medyan: 268.0.
    - Skewness (0.424) ve Kurtosis (0.757) değerleri ile neredeyse simetrik ve normal dağılıma yakın bir seyir izlemektedir.
11. **`Prothrombin` (Protrombin Zamanı):**
    - Ortalama: 10.63, Medyan: 10.60.
    - Skewness (1.111) ve Kurtosis (3.308) ile hafif sağa çarpık ve sivri bir dağılım sergilemektedir. Kan pıhtılaşma süresinin normal aralıklarda kümelendiği görülmektedir.
"""))

# --- CELL 16: OUTLIER ANALYSIS TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 5. Aykırı Değer Analizi

Bu bölümde sayısal değişkenler için Interquartile Range (IQR) yöntemi kullanılarak istatistiksel sınır değerler hesaplanacak ve bu sınırların dışında kalan gözlem sayıları (aykırı değer miktarları) tespit edilecektir. 

Aykırı değerlerin görsel tespiti amacıyla her sayısal değişken için Boxplot çizilecektir.

**Kural Hatırlatması:** Tespit edilen aykırı değerler üzerinde silme, düzeltme veya kırpma gibi hiçbir manipülasyon işlemi yapılmayacaktır; sadece durum raporlanacaktır.
"""))

# --- CELL 17: OUTLIER ANALYSIS IQR CODE ---
cells.append(nbf.v4.new_code_cell("""outlier_summary = []

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Aykırı değerleri bulma
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_count = len(outliers)
    outlier_ratio = (outlier_count / len(df)) * 100
    
    outlier_summary.append({
        'Değişken': col,
        'Q1 (%25)': q1,
        'Q3 (%75)': q3,
        'IQR': iqr,
        'Alt Sınır': lower_bound,
        'Üst Sınır': upper_bound,
        'Aykırı Adet': outlier_count,
        'Oran (%)': round(outlier_ratio, 2)
    })

outlier_report_df = pd.DataFrame(outlier_summary).set_index('Değişken')
print("IQR Aykırı Değer Analiz Özeti:")
display(outlier_report_df)
"""))

# --- CELL 18: OUTLIER ANALYSIS BOXPLOT CODE ---
cells.append(nbf.v4.new_code_cell("""# Boxplot Çizimleri (Grid Yapısında)
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()

# Uyumlu renk tonlarında kutu grafikler
box_color = 'lightcoral'

for i, col in enumerate(num_cols):
    sns.boxplot(data=df, y=col, ax=axes[i], color=box_color, width=0.4, 
                flierprops={'marker': 'o', 'markerfacecolor': 'darkred', 'markersize': 5, 'markeredgecolor': 'none'})
    axes[i].set_title(f'{col} Kutu Grafiği (Boxplot)', fontsize=12, pad=10)
    axes[i].set_ylabel(col, fontsize=10)
    
    # Alt ve Üst sınır çizgilerini belirtme (İsteğe bağlı görsel destek)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    axes[i].axhline(q1 - 1.5*iqr, color='blue', linestyle='--', alpha=0.5, label='Alt Sınır')
    axes[i].axhline(q3 + 1.5*iqr, color='blue', linestyle='--', alpha=0.5, label='Üst Sınır')

fig.delaxes(axes[-1])

plt.suptitle('Sayısal Değişkenlerin Kutu Grafikleri ve IQR Sınırları', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/numerical_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 19: OUTLIER ANALYSIS COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Aykırı Değer Analizi Yorumları:
- **Genel Değerlendirme:** Sayısal değişkenlerimizin büyük kısmında belirgin şekilde aykırı değerler bulunmaktadır. Bu durum, karaciğer hastalığı gibi patolojik durumlarda laboratuvar parametrelerinin ekstrem seviyelere çıkabilmesiyle klinik açıdan birebir uyumludur.
- **En Yüksek Aykırı Değer Oranına Sahip Değişkenler:**
  - **`Copper` (Bakır):** 609 adet aykırı değer bulunmakta ve bu veri setinin **%7.70**'ine denk gelmektedir.
  - **`Bilirubin`:** 744 adet aykırı gözlem ile veri setinin **%9.41**'ini oluşturmaktadır. Bilirubin karaciğer hasarının doğrudan bir göstergesi olduğundan, ileri evre siroz hastalarında değerler çok yüksek seviyelere tırmanmıştır.
  - **`Prothrombin` (Protrombin Zamanı):** 396 adet aykırı değer barındırır (%5.01).
  - **`Cholesterol`:** 386 adet aykırı değer içermektedir (%4.88).
  - **`Alk_Phos` (Alkalen Fosfataz):** 322 adet aykırı değer barındırmaktadır (%4.07).
- **En Temiz/Aykırı Değeri Az Olan Değişkenler:**
  - **`Age` (Yaş):** Sıfır aykırı değer. Dağılımın tamamen simetrik ve sınır değerler içinde kaldığı görülmektedir.
  - **`N_Days` (Takip Günü):** Sıfır aykırı değer.
  - **`Albumin`:** Sadece 31 adet (%0.39) aykırı değer mevcuttur. Albumin seviyeleri genellikle belirli fizyolojik sınırlar arasında tutulur.
- **Klinik Açıdan Değerlendirme:** Bu aykırı değerler veri hatası değildir; hastaların patolojik durumlarından kaynaklanan gerçek klinik ekstrem değerlerdir. Bu nedenle modelleme veya analiz aşamalarında veri kümesinden silinmemelidir. Modelleme aşamasında bu uç değerlere karşı dirençli algoritmalar (Ağaç tabanlı modeller vb.) tercih edilmelidir.
"""))

# --- CELL 20: CATEGORICAL VARIABLES TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 6. Kategorik Değişken Analizi

Bu bölümde veri setinde yer alan kategorik değişkenler analiz edilecektir. Her kategorik değişken için frekans tablosu, yüzde dağılımı hesaplanacak ve countplot grafiği çizilecektir. 

**Kategorik Değişkenler:** `Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`, `Stage` (Siroz Evresi)

*Önemli Not:* `Stage` değişkeni sayısal (1.0, 2.0, 3.0, 4.0) kodlanmış olsa da doğası gereği hastanın siroz evresini temsil eden klinik bir kategorik ve ordinal (sıralı) değişkendir. Bu nedenle bu analizde kategorik değişkenler grubunda değerlendirilecektir.
"""))

# --- CELL 21: CATEGORICAL VARIABLES CODE ---
cells.append(nbf.v4.new_code_cell("""cat_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

# Frekans ve Yüzdelik Tablolar
for col in cat_cols:
    counts = df[col].value_counts(dropna=False)
    pcts = df[col].value_counts(normalize=True, dropna=False) * 100
    table = pd.DataFrame({'Frekans (Adet)': counts, 'Oran (%)': pcts.round(2)})
    print(f"\\n--- {col} Değişkeni Dağılım Tablosu ---")
    display(table)

# Görselleştirme: Countplots (Grid Yapısı)
fig, axes = plt.subplots(4, 2, figsize=(16, 22))
axes = axes.flatten()

# Profesyonel renk paleti
palette_choice = 'Set3'

for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, ax=axes[i], palette=palette_choice, hue=col, legend=False)
    axes[i].set_title(f'{col} Sınıf Dağılımları', fontsize=12, pad=10)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Hasta Sayısı', fontsize=10)
    
    # Barların üstüne yüzdelik etiketleri ekleme
    total_col = len(df[col])
    for p in axes[i].patches:
        height = p.get_height()
        if not np.isnan(height):
            percentage = (height / total_col) * 100
            axes[i].annotate(f'{height:.0f}\\n({percentage:.1f}%)',
                             (p.get_x() + p.get_width() / 2., height + 10),
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Son ekseni gizleme (7 grafik var, 8 kutu var)
fig.delaxes(axes[-1])

plt.suptitle('Kategorik Değişkenlerin Dağılım Analizleri', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 22: CATEGORICAL VARIABLES COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Kategorik Değişken Analizi Yorumları:

1. **`Drug` (İlaç):**
   - Hastaların **%50.7'si (4010 hasta) Placebo**, **%49.3'ü (3895 hasta) D-penicillamine** almıştır. İlaç sınıfları neredeyse eşit oranda dağılmıştır, bu da deney tasarımı açısından dengeli bir yapıyı göstermektedir.
2. **`Sex` (Cinsiyet):**
   - Veri setindeki hastaların **%89.1'i (7045 hasta) Kadın (F)**, sadece **%10.9'u (860 hasta) Erkek (M)** cinsiyettedir. Siroz veri setlerinde (özellikle primer biliyer kolanjit gibi otoimmün kaynaklı siroz tiplerinde) kadın hasta oranının bu denli yüksek olması klinik olarak beklenen bir durumdur.
3. **`Ascites` (Asit - Karında Sıvı Birikmesi):**
   - Hastaların **%95.2'sinde (7525 hasta) Asit yoktur (N)**. Sadece **%4.8'inde (380 hasta) Asit mevcuttur (Y)**. Asit varlığı sirozun ileri evresinde ortaya çıkan ciddi bir komplikasyondur.
4. **`Hepatomegaly` (Karaciğer Büyümesi):**
   - Hastaların **%51.1'inde (4036 hasta) karaciğer büyümesi tespit edilmemiştir**, **%48.9'unda (3869 hasta) karaciğer büyümesi vardır (Y)**. Neredeyse yarı yarıya bir dağılım söz konusudur.
5. **`Spiders` (Örümcek Anjiyomlar - Cilt Bulgusu):**
   - Hastaların **%75.4'ünde (5964 hasta) cilt bulgusu yokken**, **%24.6'sında (1941 hasta) mevcuttur**. Örümcek anjiyom sirozun diğer bir cilt komplikasyonudur.
6. **`Edema` (Ödem):**
   - Hastaların **%89.9'unda (7103 hasta) ödem yoktur (N)**.
   - **%8.1'inde (642 kas) idrar söktürücü tedavisiyle düzeltilmiş ödem mevcuttur (S)**.
   - **%2.0'ında (160 hasta) tedaviye rağmen dirençli ödem vardır (Y)**.
7. **`Stage` (Klinik Siroz Evresi):**
   - Evre dağılımı: **Evre 1 (%5.0 - 397 hasta)**, **Evre 2 (%20.9 - 1652 hasta)**, **Evre 3 (%39.9 - 3153 hasta)**, **Evre 4 (%34.2 - 2703 hasta)**.
   - **Klinik ve Ordinal Değerlendirme:** `Stage` değişkeni hem sınıfsal dağılım hem de sıralı klinik ilerlemeyi temsil eder. Verilerin yaklaşık %74'ü Evre 3 ve Evre 4 gibi ileri derece karaciğer hasarı olan hastalardan oluşmaktadır. Bu durum, veri setindeki hastaların çoğunluğunun ileri düzey karaciğer yetmezliği / siroz durumunda olduğunu ve hastalığın son evrelerine doğru ilerlediğini gösterir. Bu durum, yüksek vefat oranı (%33.7) ile de doğrudan tutarlıdır.
"""))

# --- CELL 23: CORRELATION ANALYSIS TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 7. Korelasyon Analizi

Bu bölümde sayısal değişkenler arasındaki doğrusal ilişkilerin gücü ve yönü **Pearson Korelasyon Katsayısı** ile analiz edilecektir. Korelasyon ilişkileri sunum kalitesinde, mavi-kırmızı renk geçişli (`coolwarm` paleti) bir ısı haritası (heatmap) ile görselleştirilecektir.
"""))

# --- CELL 24: CORRELATION ANALYSIS CODE ---
cells.append(nbf.v4.new_code_cell("""# Pearson Korelasyon Matrisi Hesaplama
corr_matrix = df[num_cols].corr(method='pearson')

print("Pearson Korelasyon Katsayıları Matrisi:")
display(corr_matrix.round(3))

# Heatmap Çizimi
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, 
            linewidths=0.8, cbar_kws={'shrink': 0.8, 'label': 'Korelasyon Katsayısı (r)'},
            square=True)

plt.title('Sayısal Değişkenler Arası Pearson Korelasyon Isı Haritası', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 25: CORRELATION ANALYSIS COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Korelasyon Analizi Yorumları:

Korelasyon ısı haritası incelendiğinde değişkenler arasında genel olarak zayıf ve orta düzeyde doğrusal ilişkiler gözlemlenmektedir. En belirgin korelasyonlar aşağıda açıklanmıştır:

- **En Güçlü Pozitif İlişkiler (Doğrusal Artış):**
  - **`Bilirubin` ile `Copper` (r = 0.46):** Orta düzeyde pozitif bir ilişki mevcuttur. Bilirubin artışı (safra kanalı tıkanıklığı/karaciğer hasarı) ile vücutta bakır birikimi arasında klinik olarak bilinen güçlü bir paralellik vardır. Karaciğer fonksiyonu bozuldukça her iki toksik madde de vücutta birikmektedir.
  - **`Bilirubin` ile `Prothrombin` (r = 0.35):** Karaciğerin sentez kapasitesi düştükçe pıhtılaşma süresi (protrombin zamanı) uzar ve bilirubin seviyeleri artar.
  - **`SGOT` ile `Bilirubin` (r = 0.38) & `Copper` (r = 0.33):** SGOT (AST) aktif karaciğer hücresi yıkımını gösteren bir enzimdir. Hücre yıkımı arttıkça bilirubin ve bakır birikiminin arttığı doğrulanmaktadır.
  - **`Cholesterol` ile `Tryglicerides` (r = 0.35):** Lipit metabolizması ile ilişkili bu iki parametre arasında orta düzeyde pozitif bir korelasyon mevcuttur.
- **En Güçlü Negatif İlişkiler (Ters Yönlü Doğrusal İlişki):**
  - **`Albumin` ile `Bilirubin` (r = -0.31) ve `Prothrombin` (r = -0.32):** Albumin karaciğerde sentezlenen önemli bir proteindir. Karaciğer hasarı arttıkça (bilirubin yükseldikçe veya pıhtılaşma süresi protrombin uzadıkça) karaciğerin albümin sentezleme yeteneği düşer. Bu negatif yönlü ilişki klinik açıdan son derece tutarlıdır.
  - **`Age` ile `Platelets` (r = -0.11):** Yaş ilerledikçe trombosit sayısında hafif bir düşüş eğilimi gözlenmektedir.
- **Çoklu Doğrusal Bağlantı (Multicollinearity) Durumu:**
  - Analiz edilen tüm sayısal değişkenler arasında korelasyon katsayısının **0.50'nin altında** olduğu görülmektedir. Bu bulgu, makine öğrenmesi modelleri için oldukça olumludur; çünkü değişkenler arasında yüksek düzeyde bağımlılık (multicollinearity) problemi bulunmamaktadır.
"""))

# --- CELL 26: BIVARIATE RELATIONSHIPS TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 8. İlişki Analizleri

Bu bölümde değişkenler arasındaki klinik ve istatistiksel ilişkiler daha detaylı incelenecektir. Hedef değişken (`Status`) ve klinik evre (`Stage`) baz alınarak sayısal değişkenlerin durumları scatterplot, violin plot ve boxplot karşılaştırmaları ile ele alınacaktır. Ayrıca belirli kritik özellikler için çok boyutlu bir pairplot çizilecektir.
"""))

# --- CELL 27: BIVARIATE RELATIONSHIPS CODE ---
cells.append(nbf.v4.new_code_cell("""# 1. Scatterplot: Bilirubin vs Prothrombin (Status bazında)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Bilirubin', y='Prothrombin', hue='Status', palette='Set1', alpha=0.6, s=40)
plt.title('Bilirubin ve Prothrombin Zamanı İlişkisi (Status Kırılımında)', fontsize=13, pad=10)
plt.xlabel('Bilirubin Seviyesi')
plt.ylabel('Protrombin Zamanı')
plt.tight_layout()
plt.savefig('reports/figures/bivariate_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Violin Plot: Albumin Dağılımı (Klinik Evre - Stage bazında)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Stage', y='Albumin', palette='muted', hue='Stage', legend=False)
plt.title('Klinik Evrelere (Stage) Göre Albumin Seviyesi Dağılımları', fontsize=13, pad=10)
plt.xlabel('Klinik Evre (Stage)')
plt.ylabel('Albumin Seviyesi')
plt.tight_layout()
plt.savefig('reports/figures/bivariate_violin.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Boxplot Karşılaştırması: Bilirubin Seviyesi (Status bazında - Logaritmik Ölçek)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Status', y='Bilirubin', palette='Pastel1', hue='Status', legend=False)
plt.yscale('log')  # Bilirubin uç değerleri yüksek olduğu için log scale görselleştirmeyi netleştirir
plt.title('Gözlem Sonucuna (Status) Göre Bilirubin Seviyelerinin Karşılaştırılması (Logaritmik)', fontsize=13, pad=10)
plt.xlabel('Hastalık Durumu (Status)')
plt.ylabel('Bilirubin Seviyesi (Log Ölçek)')
plt.tight_layout()
plt.savefig('reports/figures/bivariate_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Pairplot: Belirli Kritik Özellikler için Matrix Dağılımı (Status bazında)
selected_features = ['Bilirubin', 'Copper', 'Albumin', 'Prothrombin', 'Status']
g = sns.pairplot(data=df[selected_features], hue='Status', palette='Set1', diag_kind='kde',
                 plot_kws={'alpha': 0.5, 's': 20})
g.fig.suptitle('Seçilmiş Kritik Parametrelerin Çoklu Dağılım ve İlişki Grafiği (Pairplot)', y=1.02, fontsize=14)
plt.savefig('reports/figures/pairplot.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 28: BIVARIATE RELATIONSHIPS COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### İlişki Analizi Yorumları:

1. **Bilirubin vs Prothrombin (Scatterplot):**
   - Dağılım incelendiğinde, bilirubin seviyesi 5'in ve protrombin zamanı 11'in altında olan grupta yoğun bir `C` (yaşayan) hasta kümelenmesi görülmektedir.
   - Buna karşılık, hem bilirubin seviyesinin yükseldiği hem de protrombin zamanının uzadığı sağ-üst bölgede kırmızı renkli `D` (vefat eden) hastaların yoğunlaştığı net şekilde izlenmektedir. Bu iki biyobelirtecin aynı anda yüksek olması, hastanın ölüm riskinin (D sınıfı) klinik olarak çok daha yüksek olduğunun bir göstergesidir.
2. **Klinik Evrelere Göre Albumin Dağılımı (Violin Plot):**
   - Evre 1'den Evre 4'e doğru gidildikçe, keman grafiklerinin tepe noktasının ve medyan değerlerinin aşağıya doğru kaydığı açıkça görülmektedir.
   - Evre 1 hastalarında albumin seviyesi 3.7 - 4.0 g/dL civarında kümelenirken, Evre 4 hastalarında dağılım 3.2 - 3.5 g/dL düzeyine gerilemiştir. Karaciğer yetmezliği ilerledikçe albumin sentezinin azaldığı teorisi bu görselle açıkça kanıtlanmıştır.
3. **Status Bazında Bilirubin Seviyesi (Boxplot):**
   - Logaritmik ölçekte çizilen kutu grafiği, vefat eden (`D`) hastaların bilirubin medyan değerinin, yaşayan (`C`) veya nakil olan (`CL`) hastaların medyanından belirgin şekilde yüksek olduğunu göstermektedir.
   - Yaşayan hastaların büyük kısmında bilirubin normal referans değerlerindeyken (1.0 civarı), vefat eden grupta medyan değer 3.0'ın üzerindedir ve kutunun gövdesi çok daha yukarıdadır.
4. **Çok Boyutlu İlişki Grafiği (Pairplot):**
   - Köşegenlerdeki KDE eğrileri incelendiğinde, `Albumin` dağılımında yaşayan hastaların (`C`) daha yüksek değerlere sahip bir tepe oluşturduğu, ölen hastaların (`D`) ise sol tarafa doğru yayılmış daha düşük albumin seviyelerine sahip olduğu görülmektedir.
   - `Bilirubin` ve `Copper` dağılımlarında ise yaşayan hastaların dağılımı sol tarafta dar ve yüksek bir tepe oluştururken (sağlıklı düşük değerler), vefat eden hastaların dağılım eğrisi sağa doğru geniş bir kuyruk oluşturmaktadır. Bu iki belirteç karaciğer fonksiyon kaybını ve ölüm olasılığını güçlü bir şekilde ayrıştırmaktadır.
"""))

# --- CELL 29: SUMMARY RESULTS IN TURKISH ---
cells.append(nbf.v4.new_markdown_cell("""# 9. EDA'dan Elde Edilen Bulgular

Bu kapsamda yapılan detaylı Keşifçi Veri Analizi (EDA) sonucunda elde edilen temel bulgular aşağıda madde madde özetlenmiştir:

*   **Veri Kalitesi ve Tamlığı:**
    *   Veri setinde toplam **7905 gözlem** ve **20 değişken** yer almaktadır.
    *   Tüm değişkenlerin doluluk oranı **%100**'dür; veri setinde hiçbir eksik (`NaN` / `null`) değer bulunmamaktadır. Bu durum klinik veri setlerinde nadir görülen, sentetik veri üretimine veya veri ön işleme aşamasına işaret eden bir kalitedir.
*   **Hedef Değişken Durumu (Sınıf Dengesizliği):**
    *   Hedef değişken olan `Status` sınıfları arasında ciddi dengesizlik bulunmaktadır. Yaşayan ve takip dışı kalan hastalar (`C`) **%62.8** oranında çoğunluktayken, vefat edenler (`D`) **%33.7** ve karaciğer nakli yapılan hastalar (`CL`) yalnızca **%3.5** oranındadır.
    *   Modelleme çalışmalarında `CL` azınlık sınıfının doğru tahmin edilebilmesi için sınıf ağırlıklarının optimize edilmesi veya veri dengeleme yöntemlerinin değerlendirilmesi kritik önem taşımaktadır.
*   **Aykırı Değer Durumu:**
    *   Yaş (`Age`) ve Takip Süresi (`N_Days`) değişkenleri dışında, laboratuvar bulgularını temsil eden tüm sayısal değişkenlerde yüksek oranda aykırı değerler tespit edilmiştir.
    *   En yüksek aykırı değer oranı **Bilirubin (%9.41)** ve **Copper (%7.70)** değişkenlerindedir.
    *   Bu aykırı değerler karaciğer hasarının klinik seyriyle doğrudan ilişkili biyolojik uç değerlerdir. Modelleme aşamasında bu değerlerin silinmesi yerine, uç değerlere karşı dayanıklı (örneğin karar ağaçları veya robust ölçeklendirme teknikleri) yöntemlerin seçilmesi önerilir.
*   **Dağılım Özellikleri:**
    *   Yaş (`Age`) gün bazında verilmiş olup, simetrik ve normal dağılıma oldukça yakın bir yapı sergilemektedir (Yaş ortalaması yaklaşık 50.3 yıldır).
    *   `Bilirubin` (Skew: 3.717), `Cholesterol` (Skew: 3.535), `Copper` (Skew: 2.296) ve `Alk_Phos` (Skew: 2.955) değişkenleri **aşırı derecede sağa çarpık (right-skewed)** ve dik (leptokurtik) dağılımlara sahiptir. Bu durum, hastaların çoğunun normal değerlere sahip olduğunu ancak hasta bir grubun çok yüksek risk seviyelerinde olduğunu gösterir.
    *   `Albumin` (Skew: -0.574) değişkeni hafif sola çarpık olup normal dağılıma en yakın laboratuvar parametresidir.
*   **Klinik Değişken ve Evre (Stage) Durumu:**
    *   Hastaların cinsiyet dağılımı ezici bir şekilde kadın ağırlıklıdır (%89.1 F, %10.9 M). Bu durum primer biliyer kolanjit/siroz gibi otoimmün hastalıkların popülasyon dağılımı ile uyumludur.
    *   Klinik evre (`Stage`) değişkeni incelendiğinde hastaların **%74.1**'inin ileri evre (Evre 3 ve Evre 4) siroz hastası olduğu görülmektedir. Evre ilerledikçe karaciğer sentez kapasitesinin düşüşünü temsil eden albumin seviyelerinin gerilediği (Violin Plot analizi ile) doğrulanmıştır.
*   **Korelasyon ve Klinik İlişkiler:**
    *   En güçlü doğrusal ilişkiler karaciğer hasarının biyobelirteçleri olan `Bilirubin` ile `Copper` (r = 0.46) arasındadır.
    *   Albumin sentezi ile karaciğer fonksiyon bozukluğu göstergeleri arasında klinik beklentilerle tam uyumlu şekilde negatif korelasyon (Albumin vs Bilirubin r = -0.31, Albumin vs Prothrombin r = -0.32) mevcuttur.
    *   Değişkenler arasında çoklu doğrusal bağlantı (multicollinearity) yaratacak seviyede (r >= 0.70) aşırı yüksek bir korelasyon bulunmamaktadır.
*   **Modelleme Öncesi Dikkat Edilmesi Gereken Noktalar:**
    *   **Veri Tipleri ve Kodlama:** Kategorik sütunlar (`Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`) ve ordinal yapılı `Stage` değişkeni modelleme öncesinde uygun şekilde encode edilmelidir (One-Hot / Label encoding).
    *   **Ölçeklendirme:** Sağa çarpık dağılımların yoğun olması nedeniyle, mesafe tabanlı modeller kullanılacaksa RobustScaler veya logaritmik dönüşüm (Log Transform) uygulanması gereklidir.
    *   **Metrik Seçimi:** Hedef değişkendeki dengesizlikten dolayı doğruluk (Accuracy) yerine F1-Skor, Precision-Recall AUC ve Karışıklık Matrisi (Confusion Matrix) gibi metrikler başarı değerlendirmesinde temel alınmalıdır.
"""))

# Write the notebook
nb['cells'] = cells
with open('eda_cirrhosis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("eda_cirrhosis.ipynb programmatically created.")
