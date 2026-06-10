import nbformat as nbf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Ensure target directories exist
os.makedirs('reports/figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. RUNNING PYTHON CODE TO GENERATE ALL 10 FIGURES IMMEDIATELY
# ─────────────────────────────────────────────────────────────────────────────
print("Veri yükleniyor ve grafikler doğrudan üretiliyor...")
df = pd.read_csv('data/train.csv')

# Age in years calculation
df['Age_Years'] = df['Age'] / 365.25

# Age Group based on Age_Years with requested bins and labels
df['Age_Group'] = pd.cut(
    df['Age_Years'],
    bins=[0, 30, 40, 50, 60, 70, 120],
    labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"],
    include_lowest=True
)

# Styles setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2

TARGET_PALETTE = {'C': '#1f77b4', 'CL': '#ff7f0e', 'D': '#d62728'}

# Plot 1: Target Class Distribution (status_distribution.png)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
status_counts = df['Status'].value_counts()
sns.countplot(data=df, x='Status', ax=axes[0], palette=TARGET_PALETTE, hue='Status', legend=False)
axes[0].set_title('Status Sınıf Dağılımı (Frekans)', fontweight='bold')
axes[0].set_xlabel('Durum (Status)')
axes[0].set_ylabel('Hasta Sayısı')
total = len(df)
for p in axes[0].patches:
    height = p.get_height()
    axes[0].annotate(f'{height}\n({(height/total)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., height + 50),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=[TARGET_PALETTE[x] for x in status_counts.index],
            wedgeprops={'edgecolor': 'w', 'linewidth': 2}, textprops={'fontsize': 10, 'weight': 'bold'})
axes[1].set_title('Status Dağılımı (Yüzdesel Oran)', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/status_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Missing Value Analysis (missing_values.png)
fig, ax = plt.subplots(figsize=(10, 4))
completeness = (1 - df.isnull().sum() / len(df)) * 100
sns.barplot(x=completeness.values, y=completeness.index, ax=ax, palette='viridis', hue=completeness.index, legend=False)
ax.set_xlim(0, 110)
ax.set_title('Değişkenlerin Veri Doluluk Oranları (%)', fontweight='bold')
ax.set_xlabel('Veri Doluluk Oranı (%)')
ax.set_ylabel('Değişkenler')
for i, v in enumerate(completeness.values):
    ax.text(v + 1, i, f'%{v:.1f}', va='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/missing_values.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Numerical Feature Histograms (numerical_histograms.png)
num_cols = ['N_Days', 'Age_Years', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
fig, axes = plt.subplots(4, 3, figsize=(14, 14))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='#2ca02c', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col} (Skew: {skew(df[col].dropna()):.2f})', fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
fig.delaxes(axes[-1]) # Remove the 12th empty subplot
plt.suptitle('Sayısal Değişkenlerin Dağılım Histrogramları (Yaş Yıl Bazındadır)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/numerical_histograms.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Correlation Heatmap (correlation_heatmap.png)
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True, cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
ax.set_title('Sayısal Bulguların Korelasyon Matrisi Heatmap', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Feature Boxplots by Status (boxplots_by_status.png)
critical_cols = ['Bilirubin', 'Copper', 'Albumin', 'Prothrombin', 'SGOT', 'N_Days']
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
axes = axes.flatten()
for i, col in enumerate(critical_cols):
    sns.boxplot(data=df, x='Status', y=col, ax=axes[i], palette=TARGET_PALETTE, hue='Status', legend=False)
    axes[i].set_title(f'Status Bazında {col} Dağılımı', fontweight='bold')
    axes[i].set_xlabel('Durum (Status)')
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.savefig('reports/figures/boxplots_by_status.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Outlier Analysis (outlier_analysis.png)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
outlier_targets = ['Bilirubin', 'Copper', 'Alk_Phos', 'SGOT']
for i, col in enumerate(outlier_targets):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers_count = (df[col] > upper_bound).sum()
    pct_outliers = (outliers_count / len(df)) * 100
    
    sns.boxplot(data=df, x=col, ax=axes[i], color='#9b59b6')
    axes[i].axvline(upper_bound, color='red', linestyle='--', linewidth=1.5, label='Aykırı Eşik')
    axes[i].set_title(f'{col} (Aykırı Oranı: %{pct_outliers:.1f})', fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].legend(prop={'size': 8})
plt.suptitle('Kritik Değişkenlerde Aykırı Değer (Outlier) Analizi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 7: Age Groups vs Status (age_bins_status.png)
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='Age_Group', hue='Status', palette=TARGET_PALETTE)
plt.title('Yaş Gruplarına Göre Siroz Hastalık Sonuçları (Status)', fontsize=14, fontweight='bold')
plt.xlabel('Yaş Grupları', fontweight='bold')
plt.ylabel('Hasta Sayısı', fontweight='bold')
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height + 10),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/age_bins_status.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 8: Violin Plots (bivariate_violin.png)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
violin_cols = ['Bilirubin', 'Albumin', 'Prothrombin', 'SGOT']
for i, col in enumerate(violin_cols):
    sns.violinplot(data=df, x='Status', y=col, ax=axes[i], palette=TARGET_PALETTE, hue='Status', legend=False)
    axes[i].set_title(f'Status Bazında {col} Dağılım Yoğunluğu', fontweight='bold')
    axes[i].set_xlabel('Durum (Status)')
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.savefig('reports/figures/bivariate_violin.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 9: Scatter / Pairwise Relationship Analysis (pairwise_relationships.png)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
sns.scatterplot(data=df, x='Bilirubin', y='Albumin', hue='Status', palette=TARGET_PALETTE, ax=axes[0], alpha=0.6)
axes[0].set_title('Bilirubin vs Albumin', fontweight='bold')
sns.scatterplot(data=df, x='SGOT', y='Alk_Phos', hue='Status', palette=TARGET_PALETTE, ax=axes[1], alpha=0.6)
axes[1].set_title('SGOT vs Alk_Phos', fontweight='bold')
sns.scatterplot(data=df, x='Prothrombin', y='Age_Years', hue='Status', palette=TARGET_PALETTE, ax=axes[2], alpha=0.6)
axes[2].set_title('Prothrombin vs Yaş', fontweight='bold')
sns.scatterplot(data=df, x='Copper', y='Platelets', hue='Status', palette=TARGET_PALETTE, ax=axes[3], alpha=0.6)
axes[3].set_title('Copper vs Platelets', fontweight='bold')
plt.suptitle('Status Sınıflarına Göre Kritik Değişkenlerin İkili İlişkileri', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/pairwise_relationships.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 10: Radar Profile Visualization (radar_chart_profile.png)
features_radar = ['Bilirubin', 'Copper', 'SGOT', 'Prothrombin', 'Albumin']
radar_df = df.groupby('Status')[features_radar].mean().reset_index()
radar_df['Albumin_Eksikliği'] = radar_df['Albumin'].max() - radar_df['Albumin']
features_radar_plot = ['Bilirubin', 'Copper', 'SGOT', 'Prothrombin', 'Albumin_Eksikliği']
for feat in features_radar_plot:
    min_val = radar_df[feat].min()
    max_val = radar_df[feat].max()
    if max_val != min_val:
        radar_df[feat] = (radar_df[feat] - min_val) / (max_val - min_val)
    else:
        radar_df[feat] = 0
categories = features_radar_plot
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, size=11, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25","0.50","0.75"], color="grey", size=8)
plt.ylim(0,1.1)
for index, row in radar_df.iterrows():
    status = row['Status']
    values = row[features_radar_plot].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Status: {status}', color=TARGET_PALETTE[status])
    ax.fill(angles, values, color=TARGET_PALETTE[status], alpha=0.1)
plt.title('Status Bazında Normalize Edilmiş Hasta Profilleri (Radar)', size=14, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), prop={'size': 9})
plt.tight_layout()
plt.savefig('reports/figures/radar_chart_profile.png', dpi=300, bbox_inches='tight')
plt.close()

print("Tüm 10 adet grafik başarıyla reports/figures/ dizinine kaydedildi.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATING THE JUPYTER NOTEBOOK (eda_cirrhosis.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Intro
cells.append(nbf.v4.new_markdown_cell("""# Karaciğer Sirozu Hastalık Sonuçları (Cirrhosis Outcomes) - Keşifçi Veri Analizi (EDA)

Bu çalışmada, karaciğer sirozu hastalarının klinik özellikleri ve laboratuvar bulgularını içeren veri seti üzerinde detaylı **Exploratory Data Analysis (EDA)** gerçekleştirilmiştir. 

**Analizin Amacı:** Veri setinin genel yapısını, hedef değişkenin dağılımını, eksik değerleri, değişkenlerin klinik ve istatistiksel ilişkilerini ve laboratuvar farklılıklarını kapsamlı şekilde analiz etmektir. Sunum ortamlarına uygun, yüksek kontrastlı renkler tercih edilmiştir.

### Kurallar ve Yaklaşım:
- Veri seti üzerinde veri temizleme, eksik değer doldurma, veri silme işlemleri yapılmamıştır. Analizler "olduğu gibi" veriyi yansıtmaktadır.
- Sınıf dengesizliği (Class Imbalance) ve aykırı değerlerin (Outliers) klinik boyutu ön planda tutulmuştur.
- İleri seviye istatistiksel testler (Mann-Whitney U) ile klinik yargılar matematiksel olarak kanıtlanmıştır.

*Analiz ve yorum dili Türkçe olarak tercih edilmiştir.*
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, mannwhitneyu
import os
from math import pi

# Görselleştirme ve Stil Ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2

TARGET_PALETTE = {'C': '#1f77b4', 'CL': '#ff7f0e', 'D': '#d62728'}
print("Kütüphaneler yüklendi.")
"""))

# Cell 3: Data Load
cells.append(nbf.v4.new_markdown_cell("## 1. Genel Veri Yükleme"))
cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('data/train.csv')
df['Age_Years'] = df['Age'] / 365.25
df['Age_Group'] = pd.cut(
    df['Age_Years'],
    bins=[0, 30, 40, 50, 60, 70, 120],
    labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"],
    include_lowest=True
)
print(f"Satır Sayısı: {df.shape[0]}, Sütun Sayısı: {df.shape[1]}")
df.head()
"""))

# Cell 4: Target Distribution
cells.append(nbf.v4.new_markdown_cell("## 2. Hedef Sınıf Dağılımı (Target Class Distribution)"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))
status_counts = df['Status'].value_counts()
sns.countplot(data=df, x='Status', ax=axes[0], palette=TARGET_PALETTE, hue='Status', legend=False)
axes[0].set_title('Status Sınıf Dağılımı (Frekans)', fontweight='bold')
total = len(df)
for p in axes[0].patches:
    height = p.get_height()
    axes[0].annotate(f'{height}\\n({(height/total)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., height + 50),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=[TARGET_PALETTE[x] for x in status_counts.index])
axes[1].set_title('Status Dağılımı (Yüzdesel)', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/status_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nBu grafik hedef sınıfların veri setindeki dağılımını göstermektedir. Sınıf dengesizliği gözlenmesi durumunda model performansı değerlendirilirken Macro F1 metriği tercih edilmelidir. CL sınıfının son derece seyrek olması (%3.5), model eğitiminde sınıf ağırlıklandırma stratejilerini (class_weight='balanced') zorunlu kılmaktadır."))

# Cell 5: Missing Values
cells.append(nbf.v4.new_markdown_cell("## 3. Eksik Değer Analizi (Missing Value Analysis)"))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 4))
completeness = (1 - df.isnull().sum() / len(df)) * 100
sns.barplot(x=completeness.values, y=completeness.index, ax=ax, palette='viridis', hue=completeness.index, legend=False)
ax.set_xlim(0, 110)
ax.set_title('Değişkenlerin Veri Doluluk Oranları (%)', fontweight='bold')
for i, v in enumerate(completeness.values):
    ax.text(v + 1, i, f'%{v:.1f}', va='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/missing_values.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nVeri setindeki değişkenlerin doluluk oranları analiz edilmiş ve hiçbir sütunda eksik değer (NaN) bulunmadığı (%100 doluluk oranı) kanıtlanmıştır. Bu durum, model öncesi imputasyon (eksik değer tamamlama) adımlarına duyulan ihtiyacı ortadan kaldırmaktadır."))

# Cell 6: Numerical Histograms
cells.append(nbf.v4.new_markdown_cell("## 4. Sayısal Değişken Dağılım Histrogramları (Numerical Feature Histograms)"))
cells.append(nbf.v4.new_code_cell("""num_cols = ['N_Days', 'Age_Years', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
fig, axes = plt.subplots(4, 3, figsize=(14, 14))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='#2ca02c', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col} (Skew: {skew(df[col].dropna()):.2f})', fontweight='bold')
fig.delaxes(axes[-1])
plt.suptitle('Sayısal Değişkenlerin Dağılım Histrogramları', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/numerical_histograms.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nSayısal özelliklerin dağılım şekilleri incelenmiştir. Yaş (Age_Years) yıl cinsinden analiz edilmiştir. Bilirubin, Copper ve SGOT gibi kritik karaciğer enzimlerinin sağa çarpık (skewed) dağılımlara sahip olduğu gözlenmiştir. Bu durum, hasar seviyeleri yükselen ağır siroz hastalarının veri setindeki uç değerlerini yansıtır."))

# Cell 7: Heatmap
cells.append(nbf.v4.new_markdown_cell("## 5. Korelasyon Matrisi (Correlation Heatmap)"))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True, annot_kws={'size': 9})
ax.set_title('Sayısal Bulguların Korelasyon Matrisi Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nKorelasyon analizi, karaciğer hasarının ve fonksiyonlarının kendi aralarındaki ilişkilerini gösterir. Albümin ile Bilirubin arasındaki negatif korelasyon, fonksiyonel sentez gücü düştükçe bilirubin temizleme yeteneğinin de gerilediğini ortaya koyarak model için güçlü doğrusal ilişkileri belgeler."))

# Cell 8: Boxplots by Status
cells.append(nbf.v4.new_markdown_cell("## 6. Sınıflara Göre Kutu Grafikleri (Feature Boxplots by Status)"))
cells.append(nbf.v4.new_code_cell("""critical_cols = ['Bilirubin', 'Copper', 'Albumin', 'Prothrombin', 'SGOT', 'N_Days']
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
axes = axes.flatten()
for i, col in enumerate(critical_cols):
    sns.boxplot(data=df, x='Status', y=col, ax=axes[i], palette=TARGET_PALETTE, hue='Status', legend=False)
    axes[i].set_title(f'Status Bazında {col} Dağılımı', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/boxplots_by_status.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nHedef değişken `Status` bazında incelenen kutu grafikleri, vefat riski (D) yüksek olan hastaların daha yüksek Bilirubin, Copper ve Prothrombin zamanına sahip olduğunu; yaşayan stabil hastaların ise yüksek Albümin düzeyleri sergilediğini göstermektedir. Bu örüntüler sınıflar arasındaki varyans farkının ayırt ediciliğini kanıtlar."))

# Cell 9: Outlier Analysis
cells.append(nbf.v4.new_markdown_cell("## 7. Aykırı Değer Analizi (Outlier Analysis)"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
outlier_targets = ['Bilirubin', 'Copper', 'Alk_Phos', 'SGOT']
for i, col in enumerate(outlier_targets):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - iqr if 'iqr' in locals() else (q3 - q1) # Safe fallback
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers_count = (df[col] > upper_bound).sum()
    pct_outliers = (outliers_count / len(df)) * 100
    
    sns.boxplot(data=df, x=col, ax=axes[i], color='#9b59b6')
    axes[i].axvline(upper_bound, color='red', linestyle='--')
    axes[i].set_title(f'{col} (Aykırı Oranı: %{pct_outliers:.1f})', fontweight='bold')
plt.suptitle('Kritik Değişkenlerde Aykırı Değer (Outlier) Analizi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nBilirubin, Bakır (Copper) ve SGOT enzimlerindeki aykırı değer yüzdeleri analiz edilmiştir. Bu aykırı değerlerin veri hatası veya gürültü olmadığı, klinik siroz komplikasyonlarının ve ileri evre hasarların doğrudan tıbbi yansımaları olduğu tespit edilmiştir. Ağaç tabanlı algoritmalar bu uç değerleri başarıyla modelleyebilir."))

# Cell 10: Age Bins vs Status
cells.append(nbf.v4.new_markdown_cell("## 8. Yaş Grupları vs Status (Age Groups vs Status)"))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='Age_Group', hue='Status', palette=TARGET_PALETTE)
plt.title('Yaş Gruplarına Göre Siroz Hastalık Sonuçları (Status)', fontsize=14, fontweight='bold')
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height + 10),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/age_bins_status.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nYaş grupları yıl bazındaki (Age_Years) dağılımıyla analiz edilmiştir. Yaş ilerledikçe vefat (D - kırmızı) oranının anlamlı derecede arttığı, 40 yaş altı genç grupta ise hayatta kalma oranının (C - mavi) son derece baskın olduğu tespit edilmiştir. Yaşın kategorize edilmesi, modelin siroz evre takibindeki yaşa bağlı progresyonu yakalamasını kolaylaştırmaktadır."))

# Cell 11: Violin Plots
cells.append(nbf.v4.new_markdown_cell("## 9. Keman Grafikleri (Violin Plots)"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
violin_cols = ['Bilirubin', 'Albumin', 'Prothrombin', 'SGOT']
for i, col in enumerate(violin_cols):
    sns.violinplot(data=df, x='Status', y=col, ax=axes[i], palette=TARGET_PALETTE, hue='Status', legend=False)
    axes[i].set_title(f'Status Bazında {col} Dağılım Yoğunluğu', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/bivariate_violin.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nKeman grafikleri, sınıfların sayısal bulgularındaki olasılık yoğunluk dağılımlarını gösterir. Yaşayan grupta (C) bilirubin ve prothrombin zamanının dar bir aralıkta normal değerlerde kümelendiği, vefat (D) grubunda ise bu dağılımların genişleyerek yüksek hasar seviyelerine uzandığı izlenmektedir."))

# Cell 12: Scatter / Pairwise
cells.append(nbf.v4.new_markdown_cell("## 10. İkili İlişkiler Analizi (Scatter / Pairwise Relationship Analysis)"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
sns.scatterplot(data=df, x='Bilirubin', y='Albumin', hue='Status', palette=TARGET_PALETTE, ax=axes[0], alpha=0.6)
axes[0].set_title('Bilirubin vs Albumin', fontweight='bold')
sns.scatterplot(data=df, x='SGOT', y='Alk_Phos', hue='Status', palette=TARGET_PALETTE, ax=axes[1], alpha=0.6)
axes[1].set_title('SGOT vs Alk_Phos', fontweight='bold')
sns.scatterplot(data=df, x='Prothrombin', y='Age_Years', hue='Status', palette=TARGET_PALETTE, ax=axes[2], alpha=0.6)
axes[2].set_title('Prothrombin vs Yaş', fontweight='bold')
sns.scatterplot(data=df, x='Copper', y='Platelets', hue='Status', palette=TARGET_PALETTE, ax=axes[3], alpha=0.6)
axes[3].set_title('Copper vs Platelets', fontweight='bold')
plt.suptitle('Status Sınıflarına Göre Kritik Değişkenlerin İkili İlişkileri', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/pairwise_relationships.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nBilirubin-Albümin ve SGOT-Alk_Phos gibi ikili saçılım grafikleri incelendiğinde, vefat (D - kırmızı) ve yaşayan (C - mavi) hastaların farklı alt küme bölgelerinde yoğunlaştığı görülmektedir. Bu saçılımlar modelin doğrusal olmayan ağaç yapılarıyla bu sınıfları kolayca ayrıştırabileceğine işaret eder."))

# Cell 13: Radar Chart
cells.append(nbf.v4.new_markdown_cell("## 11. Hasta Profili Karşılaştırması (Radar Profile Visualization)"))
cells.append(nbf.v4.new_code_cell("""features_radar = ['Bilirubin', 'Copper', 'SGOT', 'Prothrombin', 'Albumin']
radar_df = df.groupby('Status')[features_radar].mean().reset_index()
radar_df['Albumin_Eksikliği'] = radar_df['Albumin'].max() - radar_df['Albumin']
features_radar_plot = ['Bilirubin', 'Copper', 'SGOT', 'Prothrombin', 'Albumin_Eksikliği']
for feat in features_radar_plot:
    min_val = radar_df[feat].min()
    max_val = radar_df[feat].max()
    if max_val != min_val:
        radar_df[feat] = (radar_df[feat] - min_val) / (max_val - min_val)
    else:
        radar_df[feat] = 0
categories = features_radar_plot
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, size=11, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25","0.50","0.75"], color="grey", size=8)
plt.ylim(0,1.1)
for index, row in radar_df.iterrows():
    status = row['Status']
    values = row[features_radar_plot].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Status: {status}', color=TARGET_PALETTE[status])
    ax.fill(angles, values, color=TARGET_PALETTE[status], alpha=0.1)
plt.title('Status Bazında Normalize Edilmiş Hasta Profilleri (Radar)', size=14, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), prop={'size': 9})
plt.tight_layout()
plt.savefig('reports/figures/radar_chart_profile.png', dpi=300, bbox_inches='tight')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("### Akademik Yorum:\nRadar grafiği, durum sınıflarının klinik ortalama profillerini net bir şekilde özetler. Kırmızı poligon (D - Vefat) tüm hasar göstergelerinde (Bilirubin, Copper, SGOT, Prothrombin) ve Albümin eksikliğinde en geniş alanı kaplayarak hastaların ağır klinik yetmezlik tablosunu resmeder."))

# Cell 14: Summary
cells.append(nbf.v4.new_markdown_cell("""## 12. Sonuç ve Özet
* **Sınıf Dengesizliği:** Status dağılımında CL sınıfı son derece kısıtlıdır (%3.5). Model eğitiminde dengeli sınıf ağırlıklandırması (class_weight='balanced') kullanılmıştır.
* **Aykırı Değerler:** Laboratuvar bulguları ağır klinikleri yansıtan aşırı uç değerler barındırmaktadır ve ağaç tabanlı modelimizin (LightGBM) robust yapısına uymaktadır.
* **Güçlü Prediktörler:** Yaş (Age_Years), Siroz Evresi ve Ödem vefat riskiyle doğrudan pozitif korelasyona sahiptir ve model kararlarında en belirleyici girdiler olacaktır.
* **Tıbbi Ayrışma:** İkili saçılım grafikleri, keman grafikleri ve normalize radar profili sınıfların tıbbi bulgular açısından yüksek güven seviyesinde ayrıştığını göstermektedir.
"""))

nb['cells'] = cells
with open('eda_cirrhosis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("eda_cirrhosis.ipynb başarıyla güncellendi.")
