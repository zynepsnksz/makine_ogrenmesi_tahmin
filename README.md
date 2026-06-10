# Karaciğer Sirozu Durum Tahmini ve Klinik Karar Destek Sistemi
## 🏥 Sunum ve Tahmin Karar Destek Paneli

Bu proje, siroz hastalarının demografik, klinik semptom ve laboratuvar bulgularını kullanarak hastaların durumunu (**Kompanse/Stabil**, **Karaciğer Nakli Gerekli** veya **Vefat Riski**) tahmin etmek, kararları açıklamak ve klinisyenlere yardımcı olmak amacıyla geliştirilmiş bir klinik karar destek sistemidir.

Proje, slayt veya sunum hazırlamaya gerek kalmaksızın doğrudan Streamlit arayüzü üzerinden sunulabilecek bir **Sunum Paneli** ve **Canlı Tahmin Ekranı** içermektedir.

---

## 📂 Proje Yapısı

*   `app/streamlit_app.py`: Streamlit sunum paneli ve canlı tahmin arayüzü ana dosyası.
*   `data/`:
    *   `train.csv`: Model öncesi Keşifçi Veri Analizi (EDA) için kullanılan ham eğitim verisi.
    *   `processed_train.csv`: Ön işleme ve özellik mühendisliği adımlarından geçmiş eğitim verisi.
*   `eda_cirrhosis.ipynb`: `data/train.csv` üzerinden yapılmış, çok sınıflı sınıflandırmaya uygun, survival analizlerinden arındırılmış güncel Keşifçi Veri Analizi (EDA) Notebook'u.
*   `generate_eda.py`: `eda_cirrhosis.ipynb` notebook dosyasını otomatik üreten Python betiği.
*   `requirements.txt`: Projenin çalışması için gerekli kütüphanelerin listesi.
*   `notebooks/`: Yapılan tüm veri bilimi ve modelleme adımlarını içeren Jupyter notebookları:
    *   `02_preprocessing_feature_engineering.ipynb`
    *   `03_model_comparison.ipynb`
    *   `05_lgbm_class_weight_comparison.ipynb`
    *   `06_hyperparameter_optimization.ipynb`
    *   `08_cross_validation.ipynb`
    *   `09_learning_curve.ipynb`
    *   `10_roc_curve.ipynb`
    *   `11_shap_lime_analysis.ipynb`
*   `outputs/`: Nihai model dosyası, karşılaştırma tabloları, test scripti ve final rapor taslağı:
    *   `best_lgbm_model.pkl`: Eğitilmiş final LightGBM model nesnesi.
    *   `test_predict.py`: Tahmin hattını test eden otomatik doğrulama betiği.
    *   `final_report_draft.md`: Final sunum/PDF raporu için hazırlanmış zengin metin taslağı.
    *   `class_weight_comparison.png` & `learning_curve.png`: Sınıf ağırlıklandırma ve öğrenme eğrisi grafikleri.
    *   `roc_curve_ovr.png` & `shap_bar.png` & `shap_summary.png`: ROC eğrileri ve SHAP açıklanabilirlik grafikleri.
    *   `lime_correct_prediction.png` & `lime_wrong_prediction.png`: LIME yerel karar grafikleri.
*   `reports/figures/`: Notebook tarafından üretilen ve Streamlit'e entegre olan tüm EDA grafikleri (Mavi-Turuncu-Kırmızı kontrast stilinde):
    *   `target_distribution.png`, `missing_values.png`, `numerical_critical_distributions.png`, `numerical_distributions.png`, `numerical_boxplots.png`, `age_bins_status.png`, `categorical_vs_status.png`, `correlation_matrix.png`, `radar_chart_profile.png`, `bivariate_violin.png`.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gerekli Kütüphanelerin Kurulması
Terminal üzerinden proje kök dizininde aşağıdaki komutu çalıştırarak bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Streamlit Sunum Panelini Başlatma
Sunum ve tahmin panelini başlatmak için:
```bash
streamlit run app/streamlit_app.py
```
Uygulama başarıyla başladığında tarayıcınızda otomatik olarak `http://localhost:8501` adresinde açılacaktır.

---

## 📊 Streamlit Sunum Menüsü Akışı

Sunum panelinde sidebar aracılığıyla 11 bölüme erişilebilir:
1. **1. Proje Özeti:** Proje amacı, veri seti yapısı ve hedef sınıfların (C, CL, D) klinik anlamları.
2. **2. EDA Bulguları:** Sınıf dağılımı, eksik değerler, tahlil dağılımları ve korelasyonların sekmeli görsel sunumu. (Ayrıca yaş grupları, keman grafikleri ve klinik profilleri içeren *İleri Düzey Grafik Analizleri* sekmesi mevcuttur).
3. **3. Ön İşleme & Özellik Mühendisliği:** One-hot encoding adımları ve üretilen yeni klinik formüllerin ($\text{ALBI}$, $\text{APRI}$, $\text{BAR}$, $\text{PAI}$, $\text{Alk\_Phos\_SGOT\_Ratio}$) LaTeX biçimli gösterimi ve klinik anlamları.
4. **4. Model Karşılaştırma:** LazyPredict tarama tablosu ve LightGBM'in seçilme gerekçeleri.
5. **5. Sınıf Ağırlıklandırma Deneyleri:** CL (transplant) recall skorunu artırmak için yapılan ağırlıklandırma testleri ve confusion matrisleri.
6. **6. Hiperparametre Optimizasyonu:** RandomizedSearchCV parametreleri, nihai hiperparametre değerleri ve test seti skorları.
7. **7. Model Doğrulama (Cross-Validation):** 5-Fold Stratified CV sonuçları, öğrenme eğrisi grafiği ve overfitting yorumu.
8. **8. ROC Curve Analizi:** One-vs-Rest ROC eğrisi grafiği ve sınıf bazlı AUC değerleri.
9. **9. SHAP & LIME Açıklanabilirlik:** Global özellik önemleri (SHAP) ile tekil hasta yerel kararları (LIME) ve bunların farkı.
10. **10. Canlı Tahmin Paneli:** Sol menüden girilen klinik veriler doğrultusunda siroz durum tahmini, olasılık dağılım grafiği, hesaplanan klinik skorlar ve klinik yasal uyarılar.
11. **11. Sonuç ve Sınırlılıklar:** Akademik/klinik prototip sınırlılıkları ve genel proje kazanımları.

---

## 📝 Doğrulama ve Test Kontrol Listesi (Test Checklist)

Sistemin kararlılığını ve teslim kriterlerini doğrulamak için aşağıdaki checklist adımlarını izleyebilirsiniz:

### 1. Dosya Varlık Kontrolü
Aşağıdaki kritik çıktıların klasörlerinde bulunduğunu doğrulayın:
* [x] `outputs/best_lgbm_model.pkl` (Model varlığı)
* [x] `reports/figures/radar_chart_profile.png` (EDA klinik profil varlığı)
* [x] `reports/figures/age_bins_status.png` (EDA yaş kırılımı varlığı)
* [x] `outputs/final_report_draft.md` (PDF rapor taslağı varlığı)

### 2. Manuel Test Adımları
1. **Model Tahmin Doğrulaması:**
   Aşağıdaki komutu çalıştırarak modelin girdi kabul ettiğini ve olasılık ürettiğini doğrulayabilirsiniz:
   ```bash
   python outputs/test_predict.py
   ```
   *Beklenen Çıktı:* "Pipeline sanity checks passed! Model predictions are active and correct." mesajının konsolda görülmesi.
2. **Streamlit Görsel Arayüz Testi:**
   `streamlit run app/streamlit_app.py` komutuyla uygulamayı açın:
   * **Bölüm 2 (EDA Bulguları):** "İleri Düzey Grafik Analizleri" sekmesine tıklayın. Yaş Grupları, Normalize Profil (Radar) ve Keman (Violin) grafiklerinin hatasız yüklendiğini teyit edin.
   * **Bölüm 10 (Canlı Tahmin):** Sol taraftan değerleri değiştirerek tahmin sonucunun (Stabil, Nakil, Vefat) ve olasılık grafiklerinin dinamik olarak güncellendiğini doğrulayın.

---

## 🖨️ PDF Raporu Oluşturma
Final teslimi için hazırlanan zengin içerikli [outputs/final_report_draft.md](file:///c:/Users/Lenovo/Desktop/makine_ogrenmesi_tahmin/outputs/final_report_draft.md) dosyasını VS Code (Markdown PDF eklentisi) veya Obsidian gibi bir editör yardımıyla tek tıkla profesyonel PDF formatına dönüştürebilirsiniz.
