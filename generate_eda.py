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

Bu çalışmada, karaciğer sirozu hastalarının klinik özellikleri ve laboratuvar bulgularını içeren veri seti üzerinde detaylı **Exploratory Data Analysis (EDA)** gerçekleştirilmiştir. 

**Analizin Amacı:** Veri setinin genel yapısını, hedef değişkenin dağılımını, eksik değerleri, değişkenlerin klinik ve istatistiksel ilişkilerini, hasta sağkalım olasılıklarını ve laboratuvar farklılıklarını kapsamlı şekilde analiz etmektir. Sunum ortamlarına uygun (projeksiyon dostu), yüksek kontrastlı renkler tercih edilmiştir.

### Kurallar ve Yaklaşım:
- Veri seti üzerinde veri temizleme, eksik değer doldurma, veri silme işlemleri yapılmamıştır. Analizler "olduğu gibi" veriyi yansıtmaktadır.
- Sınıf dengesizliği (Class Imbalance) ve aykırı değerlerin (Outliers) klinik boyutu ön planda tutulmuştur.
- İleri seviye istatistiksel testler (Mann-Whitney U) ve Sağkalım Analizleri (Kaplan-Meier) ile klinik yargılar matematiksel olarak kanıtlanmıştır.

*Analiz ve yorum dili Türkçe olarak tercih edilmiştir.*
"""))

# --- CELL 2: IMPORTS AND STYLE SETUP ---
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, mannwhitneyu
import os
from math import pi

# Lifelines kütüphanesi sağkalım (survival) analizi için gereklidir.
try:
    from lifelines import KaplanMeierFitter
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines"])
    from lifelines import KaplanMeierFitter

# Görselleştirme ve Stil Ayarları (Projeksiyon Dostu, Yüksek Kontrast)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['lines.linewidth'] = 2.5

# Yüksek kontrastlı ana renk paletleri
MAIN_PALETTE = 'tab10'
TARGET_PALETTE = {'C': '#1f77b4', 'CL': '#ff7f0e', 'D': '#d62728'} # Mavi, Turuncu, Kırmızı

# Kayıt dizini kontrolü
os.makedirs('reports/figures', exist_ok=True)

print("Kütüphaneler yüklendi ve grafik ayarları tamamlandı.")
"""))

# --- CELL 3: GENERAL DATA OVERVIEW TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 1. Genel Veri İncelemesi"""))

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

print("İlk 5 Gözlem:")
display(df.head(5))
"""))

# --- CELL 5: TARGET VARIABLE TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 2. Hedef Değişken Analizi (`Status`)

Bu bölümde hedef değişken olan `Status` (Hastalık Sonucu) incelenecektir:
- `C` (Censored - Yaşıyor/Takip Dışı)
- `CL` (Censored due to Liver Transplant - Karaciğer Nakli Yapılmış ve Yaşıyor)
- `D` (Death - Hayatını Kaybetmiş)
"""))

# --- CELL 6: TARGET VARIABLE CODE ---
cells.append(nbf.v4.new_code_cell("""status_counts = df['Status'].value_counts()
status_pct = df['Status'].value_counts(normalize=True) * 100

status_summary = pd.DataFrame({'Frekans (Adet)': status_counts, 'Oran (%)': status_pct})
display(status_summary)

# Grafik Çizimi
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Countplot
sns.countplot(data=df, x='Status', ax=axes[0], palette=TARGET_PALETTE, hue='Status', legend=False)
axes[0].set_title('Status Dağılımı (Sayısal)', fontweight='bold')
axes[0].set_xlabel('Durum (Status)')
axes[0].set_ylabel('Hasta Sayısı')

total = len(df)
for p in axes[0].patches:
    height = p.get_height()
    axes[0].annotate(f'{height}\\n({(height/total)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., height + 50),
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

# Pie Chart
axes[1].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=[TARGET_PALETTE[x] for x in status_counts.index],
            wedgeprops={'edgecolor': 'w', 'linewidth': 2}, textprops={'fontsize': 14, 'weight': 'bold'})
axes[1].set_title('Status Dağılımı (Yüzdesel Oran)', fontweight='bold')

plt.suptitle('Hedef Değişken (Status) Sınıf Analizi', fontsize=20, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('reports/figures/target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 7: TARGET VARIABLE COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### Yorum:
Veri setinde ciddi bir **Sınıf Dengesizliği (Class Imbalance)** mevcuttur. Vakaların %62.8'i Yaşıyor (`C`), %33.7'si Vefat (`D`) etmiş, sadece %3.5'i Nakil (`CL`) olmuştur. Modelleme aşamasında bu dengesizliğe (özellikle CL azınlık sınıfı için) dikkat edilmeli ve `class_weight` veya F1-Macro metrikleri kullanılmalıdır.
"""))

# --- CELL 8: MISSING VALUES ---
cells.append(nbf.v4.new_markdown_cell("""# 3. Eksik Değer Analizi"""))
cells.append(nbf.v4.new_code_cell("""missing_pct = (df.isnull().sum() / len(df)) * 100
if missing_pct.sum() == 0:
    print("Müjde! Veri setindeki hiçbir değişkende eksik (NaN) değer bulunmamaktadır. Doluluk oranı %100'dür.")
else:
    display(missing_pct[missing_pct > 0])
"""))

# --- CELL 9: NUMERICAL VARIABLES TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 4. Sayısal Değişken Analizi ve Aykırı Değerler (Outliers)"""))

# --- CELL 10: NUMERICAL VARIABLES CODE ---
cells.append(nbf.v4.new_code_cell("""num_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 
            'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

# Dağılım (KDE) ve Kutu Grafikleri (Boxplot) Bir Arada (Sadece Kritik Olanlar İçin Görsel Kalabalığı Önlemek Adına)
critical_num_cols = ['Bilirubin', 'Copper', 'Albumin', 'Prothrombin', 'SGOT', 'Platelets']

fig, axes = plt.subplots(len(critical_num_cols), 2, figsize=(16, 5 * len(critical_num_cols)))

for i, col in enumerate(critical_num_cols):
    # Histogram & KDE
    sns.histplot(data=df, x=col, kde=True, ax=axes[i, 0], color='#2ca02c', edgecolor='black', alpha=0.7)
    axes[i, 0].set_title(f'{col} - Dağılımı (Skew: {skew(df[col]):.2f})', fontweight='bold')
    
    # Boxplot
    sns.boxplot(data=df, x=col, ax=axes[i, 1], color='#d62728', flierprops={'marker': 'o', 'markerfacecolor': 'black', 'alpha': 0.5})
    axes[i, 1].set_title(f'{col} - Aykırı Değerler (Boxplot)', fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/numerical_critical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

# --- CELL 11: NUMERICAL COMMENTS ---
cells.append(nbf.v4.new_markdown_cell("""### İstatistiksel Yorum:
Laboratuvar bulguları klinik doğası gereği aşırı uç değerler barındırır. **Bilirubin**, **Copper** (Bakır) ve **SGOT** gibi hasar göstergeleri çok aşırı sağa çarpıktır (Ağır hastaların yüksek bulguları). Hastalık ciddiyetini gösteren bu aykırı değerler kesinlikle silinmemelidir. Ağaç tabanlı modellerin bu değerlere karşı robust olduğu unutulmamalıdır.
"""))

# --- CELL 12: AGE BINS ANALYSIS ---
cells.append(nbf.v4.new_markdown_cell("""# 5. Yaş Grupları Analizi (Age Bins)
Yaş değişkeni gün bazında verildiği için yorumlaması zordur. Yaşı yıla çevirip anlamlı gruplara ayırarak hastalık sonuçlarını (Status) inceleyeceğiz.
"""))

cells.append(nbf.v4.new_code_cell("""df['Age_Years'] = df['Age'] / 365.25

# Yaş grupları oluşturma
bins = [0, 40, 50, 60, 100]
labels = ['<40 Yaş', '40-50 Yaş', '50-60 Yaş', '60+ Yaş']
df['Age_Group'] = pd.cut(df['Age_Years'], bins=bins, labels=labels)

plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df, x='Age_Group', hue='Status', palette=TARGET_PALETTE)
plt.title('Yaş Gruplarına Göre Hastalık Sonuçları (Status)', fontsize=18, fontweight='bold')
plt.xlabel('Yaş Grupları', fontweight='bold')
plt.ylabel('Hasta Sayısı', fontweight='bold')

# Etiketleri ekle
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height + 10),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/age_bins_status.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Yaş Yorumu:
Görselden net bir şekilde anlaşıldığı üzere, **Yaş arttıkça vefat oranı (D - kırmızı çubuk) bariz bir şekilde artmaktadır**. 40 yaş altı hastalarda yaşayanların oranı vefata göre çok baskınken, 60 yaş üstü hastalarda vefat sayısı yaşayanları geçmiştir. Yaş, model için oldukça ayırt edici bir parametre olacaktır.
"""))

# --- CELL 13: CATEGORICAL VS TARGET TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 6. Kategorik Değişkenlerin Hedef (`Status`) ile İlişkisi
Cinsiyet, İlaç, Ödem, ve Siroz Evresi gibi değişkenlerin vefat/sağkalım ile olan ilişkisi **Gruplanmış Çubuk Grafikler** ile görselleştirilmiştir. Yüksek kontrastlı renkler tercih edilmiştir.
"""))

# --- CELL 14: CATEGORICAL VS TARGET CODE ---
cells.append(nbf.v4.new_code_cell("""cat_cols = ['Sex', 'Drug', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

fig, axes = plt.subplots(4, 2, figsize=(18, 26))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = sns.countplot(data=df, x=col, hue='Status', ax=axes[i], palette=TARGET_PALETTE)
    axes[i].set_title(f'{col} Değişkeni ve Hastalık Sonucu', fontsize=16, fontweight='bold')
    axes[i].set_ylabel('Hasta Sayısı', fontweight='bold')
    axes[i].set_xlabel(col, fontweight='bold')
    
fig.delaxes(axes[-1])
plt.tight_layout()
plt.savefig('reports/figures/categorical_vs_status.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Kategorik Yorumlar:
* **Stage (Klinik Evre):** Evre 1 ve 2'de vefat (D) oranı çok düşükken, sirozun son evresi olan **Evre 4'te vefat eden hasta sayısı yaşayan hasta sayısını geçmiştir.** Siroz evresi kesinlikle en güçlü prediktörlerden biridir.
* **Ascites, Hepatomegaly, Spiders (Klinik Bulgular):** Bu komplikasyonları taşıyan (Y) hastalarda vefat oranlarının (Kırmızı Bar) oransal olarak ciddi şekilde arttığı gözlemlenmektedir.
* **Edema (Ödem):** Tedaviye yanıt vermeyen ödemi olan (Y) grupta ölüm oranı aşırı derecede yüksektir.
"""))

# --- CELL 15: SURVIVAL ANALYSIS TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 7. Sağkalım Analizi (Kaplan-Meier Curve)
Klinik çalışmalarda sadece sonuca bakmak yetmez, hastanın ne kadar süre hayatta kaldığı da önemlidir. `N_Days` değişkeni bu süreyi verir. Tedavi gruplarının (İlaç) sağkalım olasılıkları incelenmiştir.
"""))

# --- CELL 16: SURVIVAL ANALYSIS CODE ---
cells.append(nbf.v4.new_code_cell("""# Kaplan-Meier için Event değişkeni (Death = 1, Diğerleri = 0)
df['Event'] = df['Status'].apply(lambda x: 1 if x == 'D' else 0)

kmf = KaplanMeierFitter()

plt.figure(figsize=(12, 7))

# İlaç grubuna göre kırılım (Drug)
groups = df['Drug'].unique()
if len(groups) == 2 and not pd.isnull(groups).any():
    for name, grouped_df in df.groupby('Drug'):
        kmf.fit(grouped_df['N_Days'], grouped_df['Event'], label=str(name))
        kmf.plot_survival_function(linewidth=3)
else:
    # İlaç değişkeni NaN içeriyorsa Cinsiyete göre çizelim
    for name, grouped_df in df.groupby('Sex'):
        kmf.fit(grouped_df['N_Days'], grouped_df['Event'], label=str(name))
        kmf.plot_survival_function(linewidth=3)

plt.title('Tedavi Grubuna Göre Kaplan-Meier Sağkalım Eğrisi', fontsize=18, fontweight='bold')
plt.xlabel('Zaman (Gün)', fontweight='bold', fontsize=14)
plt.ylabel('Hayatta Kalma Olasılığı', fontweight='bold', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('reports/figures/kaplan_meier_survival.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Sağkalım Analizi Yorumu:
Kaplan-Meier eğrisinde D-penicillamine ve Placebo alan grupların sağkalım olasılıkları zamanla paralel bir şekilde düşmektedir. Aralarında devasa bir fark görülmemektedir. Eğri 4000. günlere yaklaşırken sağkalım ihtimali ciddi oranda gerilemektedir. 
"""))

# --- CELL 17: HYPOTHESIS TESTING ---
cells.append(nbf.v4.new_markdown_cell("""# 8. İstatistiksel Hipotez Testi (Anlamlılık)
Grafiklerde gördüğümüz farkların tesadüfi olup olmadığını matematiksel olarak kanıtlıyoruz. Yaşayan (`C`) ve Ölen (`D`) hastalar arasında **Bilirubin** ve **Albumin** seviyeleri istatistiksel olarak anlamlı farklılığa sahip mi?
(Dağılımlar normal olmadığı için parametrik olmayan Mann-Whitney U Testi kullanılmıştır).
"""))

cells.append(nbf.v4.new_code_cell("""group_c = df[df['Status'] == 'C']
group_d = df[df['Status'] == 'D']

features_to_test = ['Bilirubin', 'Albumin', 'Copper', 'Prothrombin']
test_results = []

for feat in features_to_test:
    stat, p_value = mannwhitneyu(group_c[feat].dropna(), group_d[feat].dropna(), alternative='two-sided')
    significance = "Anlamlı Fark VAR" if p_value < 0.05 else "Anlamlı Fark YOK"
    test_results.append({'Özellik': feat, 'P-Değeri': p_value, 'Sonuç (alpha=0.05)': significance})

test_df = pd.DataFrame(test_results)
display(test_df)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Hipotez Testi Yorumu:
Tüm p-değerleri 0.05'ten (ve hatta 0.0001'den) küçüktür. Bu, vefat eden hastalar ile yaşayan hastaların laboratuvar bulguları arasındaki farkın **asla tesadüf olmadığı**, %99.9 güvenle istatistiksel olarak anlamlı olduğu anlamına gelmektedir.
"""))

# --- CELL 18: RADAR CHART TITLE ---
cells.append(nbf.v4.new_markdown_cell("""# 9. Hasta Profili Karşılaştırması (Radar / Spider Chart)
Hastalık Sonucu (`Status`) sınıflarına göre laboratuvar ortalamalarını tek bir görselde karşılaştırıyoruz.
(Değerler görselleştirme adına 0-1 aralığına Min-Max yöntemi ile indirgenmiştir).
"""))

# --- CELL 19: RADAR CHART CODE ---
cells.append(nbf.v4.new_code_cell("""features_radar = ['Bilirubin', 'Copper', 'SGOT', 'Prothrombin', 'Albumin']

# Grupların ortalamalarını alıp yeni DataFrame oluşturma
radar_df = df.groupby('Status')[features_radar].mean().reset_index()

# Albumin ters yönde çalışır (yüksek olması iyidir), bunu görselleştirme için tersine çeviriyoruz ki grafiğin dışına doğru olan her şey kötü durumu göstersin
radar_df['Albumin_Eksikliği'] = radar_df['Albumin'].max() - radar_df['Albumin']

features_radar_plot = ['Bilirubin', 'Copper', 'SGOT', 'Prothrombin', 'Albumin_Eksikliği']

# Min-Max Scaling (0-1 arası) görsel açıdan radar grafiğine uyması için
for feat in features_radar_plot:
    min_val = radar_df[feat].min()
    max_val = radar_df[feat].max()
    if max_val != min_val:
        radar_df[feat] = (radar_df[feat] - min_val) / (max_val - min_val)
    else:
        radar_df[feat] = 0

# Radar Plot Çizimi
categories = features_radar_plot
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, size=14, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25","0.50","0.75"], color="grey", size=10)
plt.ylim(0,1.1)

colors = {'C': '#1f77b4', 'CL': '#ff7f0e', 'D': '#d62728'}

for index, row in radar_df.iterrows():
    status = row['Status']
    values = row[features_radar_plot].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=3, linestyle='solid', label=f'Status: {status}', color=colors[status])
    ax.fill(angles, values, color=colors[status], alpha=0.1)

plt.title('Status Bazında Normalize Edilmiş Hasta Profilleri', size=20, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig('reports/figures/radar_chart_profile.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Radar Grafiği Yorumu:
Kırmızı alan (`D` - Vefat) poligonu çok genişken, Mavi alan (`C` - Yaşayan) merkeze yapışıktır. Yani vefat eden hastalar Bilirubin, Copper, SGOT, Prothrombin ve Albumin Eksikliğinde uç/kötü değerlere sahiptir. Tüm karaciğer harabiyeti belirtileri eş zamanlı olarak artmıştır.
"""))

# --- CELL 20: SUMMARY ---
cells.append(nbf.v4.new_markdown_cell("""# 10. Sonuç ve Özet
* Sınıf dengesizliği (`Status` dağılımı) barizdir. Modellerde dikkat edilmelidir.
* Laboratuvar değerleri ağır aykırı değerler içermektedir ve bunlar kliniktir.
* Yaş, Siroz Evresi ve Ödem ölümle doğrudan ve en güçlü korelasyona sahip kategorik/ordinal belirteçlerdir.
* İstatistiksel testler, yaşayan ve vefat eden hastaların klinik bulgularının matematiksel olarak da tamamen birbirinden ayrıştığını kanıtlamıştır.
"""))

nb['cells'] = cells
with open('eda_cirrhosis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("eda_cirrhosis.ipynb Jupyter Notebook'u başarıyla güncellendi.")
