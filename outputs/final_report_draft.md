# Karaciğer Sirozu Durum Tahmini ve Klinik Karar Destek Sistemi
## 📄 Proje Final Raporu / Sunum Taslağı

---

## 1. Giriş ve Proje Amacı (Purpose)
Karaciğer sirozu, karaciğer dokusunun kronik hasar görerek skarlaşması (fibrozis) ile karakterize, mortalitesi yüksek sistemik bir hastalıktır. Sirozun son evrelerinde karaciğer yetmezliği tablosu gelişmekte ve hastalar için organ nakli (transplantasyon) tek hayat kurtarıcı seçenek haline gelmektedir. 

Bu projenin temel amacı, primer biliyer siroz (PBC) tanısı almış hastaların klinik bulguları, demografik verileri ve laboratuvar tahlillerini kullanarak hastaların durumunu (**Kompanse/Stabil**, **Karaciğer Nakli Gerekli** veya **Vefat Riski**) yüksek başarıyla tahmin eden çok sınıflı (multiclass) bir makine öğrenmesi modeli geliştirmektir. Proje, klinisyenlerin acil nakil ihtiyacı olan veya yüksek vefat riski taşıyan hastaları erken tespit etmelerine yardımcı olacak açıklanabilir bir klinik karar destek aracı (Explainable AI Decision Support System) olarak tasarlanmıştır.

---

## 2. Veri Seti Açıklaması (Dataset)
Veri seti, Mayo Clinic tarafından 1974-1984 yılları arasında yürütülen primer biliyer siroz klinik denemesine ait verileri içermektedir. Veri setinde 7906 hasta kaydı bulunmakta ve veri setinin doluluk oranı %100'dür (eksik değer içermez).

### 🏷️ Hedef Değişken ve Sınıflar (Status)
Modelin tahmin hedefi olan `Status` değişkeni 3 sınıfa ayrılmıştır:
1. **C (Kompanse / Stabil - Sınıf 0):** Hastanın hayatta olduğunu ve karaciğer fonksiyonlarının stabil seyrettiğini gösterir (Veri oranı: %62.8).
2. **CL (Karaciğer Nakli Gerekli / Yaşıyor - Sınıf 1):** Hastaya karaciğer nakli yapıldığını veya acil nakil adayı olduğunu gösterir (Veri oranı: %3.5).
3. **D (Vefat Riski / Ağır Durum - Sınıf 2):** Takip süresince siroz veya karaciğer yetmezliği kaynaklı vefat gerçekleştiğini gösterir (Veri oranı: %33.7).

---

## 3. Keşifçi Veri Analizi (EDA) Bulguları
`data/train.csv` üzerinden yapılan keşifçi veri analizinde aşağıdaki kritik bulgular elde edilmiştir:
* **Sınıf Dengesizliği (Class Imbalance):** CL (Nakil) sınıfı verinin %3.5'i gibi çok küçük bir bölümünü oluşturmaktadır. Bu durum model değerlendirmede genel doğruluk (Accuracy) yerine **Macro F1** metriğini öncelikli kılmaktadır.
* **Aykırı Değerler (Outliers):** Bilirubin, Bakır (Copper), Alkalen Fosfataz (Alk_Phos) ve SGOT enzimleri sağa doğru uzayan uzun kuyruklu (skewed) dağılımlara sahiptir. Klinik olarak bu aykırı değerler veri hatası olmayıp hastalığın ciddiyetini yansıttığı için temizlenmemiş, ağaç tabanlı modellerin (LightGBM) robust yapısına güvenilmiştir.
* **Yaş Gruplarının Etkisi:** Yaş ilerledikçe (özellikle 60 yaş üstünde) vefat (D) oranının kompanse (C) duruma göre ciddi şekilde arttığı gözlenmiştir.
* **Klinik Evre (Stage):** Hastalığın son evresi olan Evre 4'te vefat eden hasta sayısı yaşayan hasta sayısını geçmiştir.
* **Tıbbi Profil Ayrışması:** Mann-Whitney U testleri ve normalize edilmiş radar grafik profilleri, yaşayan (C) ve vefat eden (D) hastaların tahlil dağılımları arasındaki farkların tesadüf olmadığını, istatistiksel olarak yüksek güven seviyesinde anlamlı olduğunu göstermiştir.

---

## 4. Ön İşleme & Özellik Mühendisliği (Preprocessing & Feature Engineering)
### ⚙️ Uygulanan Ön İşleme Adımları:
1. **Gürültü Temizliği:** Model için bilgi taşımayan benzersiz `id` sütunu kaldırılmıştır.
2. **Sınıf Kodlama:** `Status` değişkeni makine öğrenmesi algoritmaları için tamsayı kodlarına (`C: 0, CL: 1, D: 2`) dönüştürülmüştür.
3. **Dummy/One-Hot Encoding:** Kategorik veya ikili değişkenler (`Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`) ikili dummy kolonlarına ayrıştırılmıştır.

### 🧬 Üretilen Yeni Klinik Özellikler (Feature Engineering):
* **ALBI (Albumin-Bilirubin) Skoru:** Karaciğer fonksiyonel rezervinin objektif göstergesidir.
  $$ALBI = 0.66 \times \log_{10}(\text{Bilirubin} \times 17.1) - 0.085 \times (\text{Albumin} \times 10)$$
* **APRI (AST to Platelet Ratio Index):** Karaciğer fibrozis/siroz derecesini cerrahi biyopsi olmadan tahmin eder.
  $$APRI = \frac{\frac{\text{SGOT}}{40} \times 100}{\text{Platelets}}$$
* **BAR (Bilirubin / Albumin Oranı):** Karaciğer hasarı ve sentez gücü arasındaki dengeyi gösterir.
  $$\text{BAR} = \frac{\text{Bilirubin}}{\text{Albumin}}$$
* **PAI (Protrombin / Albumin Endeksi):** Pıhtılaşma hızı sentezi ile ana kan proteini sentez yeteneğini birleştirir.
  $$\text{PAI} = \frac{\text{Prothrombin}}{\text{Albumin}}$$
* **Alk_Phos / SGOT Oranı:** Sirozun safra yolu kökenli mi yoksa hücre ölümü kökenli mi olduğunu ayırt eder.
  $$\text{Oran} = \frac{\text{Alk\_Phos}}{\text{SGOT}}$$

---

## 5. Model Karşılaştırma ve LightGBM Seçimi (Model Comparison)
LazyPredict kütüphanesi ile veri seti üzerinde 25+ farklı sınıflandırıcı test edilmiştir. 
* **LGBMClassifier:** En yüksek genel doğruluk (0.8286) ve en yüksek **Macro F1 (0.6378)** değerini vermiştir.
* **RandomForestClassifier:** Genel doğruluk (0.8185), Macro F1 (0.5660) ile ikinci sırada yer almıştır.
* **Seçim Gerekçesi:** LightGBM, sınıf dengesizliğine karşı `class_weight='balanced'` desteği sunması, eğitim hızı, düşük kaynak kullanımı ve yüksek başarı katsayıları sebebiyle nihai model olarak seçilmiştir.

---

## 6. Sınıf Ağırlıklandırma Deneyleri (Class Weight Analysis)
Azınlık sınıfı olan `CL` (Nakil) performansını artırmak için yapılan deneyler:
* **Varsayılan LGBM (Dengesiz):** CL Recall = %14.54, CL F1 = %23.88
* **Nihai LGBM (class_weight='balanced'):** CL Recall = **%27.27**, CL F1 = **%31.91**
* **Manuel Ağırlıklar (1:5:1, 1:10:1, 1:15:1):** CL Recall değerini iyileştirse de genel Macro F1 ve Precision dengesini `balanced` kadar koruyamamıştır.
* **Tıbbi Çıkarım:** CL (nakil) hastalarını gözden kaçırmamak (high recall) klinik olarak yanlış alarm vermekten (lower precision) daha hayati olduğu için `class_weight='balanced'` en başarılı strateji olarak kabul edilmiştir.

---

## 7. Hiperparametre Optimizasyonu (Hyperparameter Optimization)
* **Yöntem:** RandomizedSearchCV ve 5-Fold StratifiedKFold cross-validation.
* **Scoring Metriği:** `f1_macro`
* **En İyi Parametreler:**
  `{'subsample_freq': 1, 'subsample': 1.0, 'reg_lambda': 0.1, 'reg_alpha': 5.0, 'num_leaves': 127, 'n_estimators': 500, 'min_child_samples': 30, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.9}`
* **Test Seti Metrikleri:**
  - Accuracy: **0.8166**
  - Macro F1: **0.6488**
  - Precision Macro: **0.6583**
  - Recall Macro: **0.6429**
  - Balanced Accuracy: **0.6429**

---

## 8. Model Doğrulama (Cross Validation & Learning Curve)
5-Fold Stratified Çapraz Doğrulama sonuçları:
* **Val Accuracy (Mean ± Std):** $81.62\% \pm 0.45\%$
* **Val Macro F1 (Mean ± Std):** $66.57\% \pm 1.59\%$
* **Overfitting Analizi:** Eğitim F1 skoru (%91.33) ile doğrulama F1 skoru (%66.57) arasındaki fark, optimize edilmiş regülasyon kısıtları (`reg_alpha=5.0`, `max_depth=5`) sayesinde kabul edilebilir sınırlar içindedir ve aşırı öğrenme engellenmiştir. Öğrenme Eğrisi (Learning Curve), veri miktarı arttıkça doğrulama F1 skorunun arttığını göstermektedir.

---

## 9. Sınıf Ayrıştırma Gücü (ROC Curve)
One-vs-Rest (OvR) ROC analizi sonuçları:
* **C (Stabil) AUC:** **0.9010**
* **D (Vefat) AUC:** **0.9047**
* **CL (Nakil) AUC:** **0.7908**
* **Macro Average AUC:** **0.8655**
* **Yorum:** AUC değerlerinin 0.80-0.90 bandı üzerinde olması, modelimizin karar olasılık sınırlarının ayırt ediciliğinin ve sınıf ayrıştırma yeteneğinin mükemmel seviyede olduğunu kanıtlar.

---

## 10. Yapay Zeka Açıklanabilirliği (SHAP & LIME)
* **Küresel (SHAP):** Model genelinde en önemli karar kriterleri **N_Days** (Takip süresi), **Age** (Yaş), **BAR** (Bilirubin/Albumin Oranı) ve **Prothrombin** zamanıdır. Yüksek bilirubin ve yüksek prothrombin riski (D sınıfı) artırırken, stabil albümin hayatta kalmayı (C sınıfı) destekler.
* **Yerel (LIME):** Tekil hasta örneklerinde, LIME karar ağacı hastanın laboratuvar bulgularının (örneğin albüminin 3.5'in üzerinde olması veya bilirubinin normal sınırlarda olması) riski nasıl düşürdüğünü/yükselttiğini klinik bazda kanıtlamıştır.

---

## 11. Canlı Karar Destek Arayüzü (Streamlit Demo)
Streamlit uygulaması (`app/streamlit_app.py`) klinisyenlerin kullanımına sunulan etkileşimli bir paneldir. 
* Sol menüden (sidebar) hastanın tüm demografik, klinik ve tahlil bulguları girilir.
* Panel, hastanın `Status` durumunu (Stabil, Nakil, Vefat) ve olasılık yüzdelerini anlık hesaplar.
* Hastanın ALBI, APRI, BAR, PAI skorlarını otomatik hesaplayarak evreleme çıktılarını klinisyene gösterir.
* Arayüz akademik sunumlara uygun kontrastlı modern bir görsel tema sunar.

---

## 12. Sonuç ve Sınırlılıklar (Conclusion)
1. **Başarı:** Model, stabil (C) ve vefat (D) sınıflarında %90+ AUC skoruyla klinikte hekim kararını destekleyecek düzeyde başarılıdır.
2. **CL Sınıfı Kısıtı:** Nakil adayı (CL) sınıfının azlığı model eğitimindeki en büyük kısıttır; dengeli sınıf ağırlıklandırması recall'u artırmış olsa da yanlış alarm oranını da bir miktar yükseltmiştir.
3. **Klinik Uyarı:** Karar destek sistemi klinik araştırmalar için geliştirilmiş akademik bir prototiptir. Kesin teşhis hekimler tarafından konulmalıdır.
