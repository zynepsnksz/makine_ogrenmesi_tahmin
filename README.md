# Karaciğer Sirozu Durum Tahmini ve Klinik Karar Destek Sistemi

Bu proje, siroz hastalarının demografik, klinik ve laboratuvar parametrelerini kullanarak hastaların durumunu (**Kompanse/Stabil**, **Karaciğer Nakli Gerekli** veya **Vefat Riski**) tahmin etmek, kararları açıklamak ve klinisyenlere yardımcı olmak amacıyla geliştirilmiş bir klinik karar destek sistemidir.

---

## 🚀 Streamlit Uygulaması Çalıştırma Talimatı

Uygulamayı yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Gerekli Kütüphanelerin Kurulması:**
    Gerekli tüm paketlerin (LightGBM, Scikit-Learn, Streamlit, SHAP, LIME, Matplotlib vb.) sisteminizde kurulu olduğundan emin olun:
    ```bash
    pip install streamlit lightgbm scikit-learn shap lime matplotlib seaborn pandas numpy
    ```

2.  **Uygulamayı Başlatma:**
    Terminal üzerinden projenin kök dizinine (workspace) gidin ve aşağıdaki komutu çalıştırın:
    ```bash
    streamlit run app/streamlit_app.py
    ```

3.  **Tarayıcıda Görüntüleme:**
    Uygulama başarıyla başlatıldığında tarayıcınızda otomatik olarak yeni bir sekme açılacaktır (Genellikle `http://localhost:8501` adresinde).

---

## 📂 Proje Yapısı

*   `app/streamlit_app.py`: Streamlit uygulaması ana dosyası.
*   `data/processed_train.csv`: Ön işleme ve özellik mühendisliği adımlarından geçmiş eğitim verisi.
*   `notebooks/`: Yapılan tüm veri bilimi ve modelleme adımlarını içeren Jupyter notebookları:
    *   `02_preprocessing_feature_engineering.ipynb`
    *   `03_model_comparison.ipynb`
    *   `05_lgbm_class_weight_comparison.ipynb`
    *   `06_hyperparameter_optimization.ipynb`
    *   `08_cross_validation.ipynb`
    *   `09_learning_curve.ipynb`
    *   `10_roc_curve.ipynb`
    *   `11_shap_lime_analysis.ipynb`
*   `outputs/`: Final model ikili dosyası, karşılaştırma tabloları, metrikler ve görselleştirmeler:
    *   `best_lgbm_model.pkl`: Eğitilmiş final model nesnesi.
    *   `class_weight_comparison.png`: Sınıf ağırlıklandırma hat matrisleri karşılaştırması.
    *   `learning_curve.png`: Model öğrenme eğrisi grafiği.
    *   `roc_curve_ovr.png`: Sınıf ayrıştırma gücünü gösteren ROC eğrileri.
    *   `shap_summary.png` & `shap_bar.png`: SHAP global açıklanabilirlik grafikleri.
    *   `lime_correct_prediction.png` & `lime_wrong_prediction.png`: LIME yerel karar grafikleri.

---

## 🧬 Özellik Mühendisliği (Feature Engineering)

Projede karaciğer rezervini ve hasarını daha iyi yansıtmak adına literatürde kabul görmüş 5 klinik skor hesaplanmaktadır:

1.  **ALBI Skoru:** Karaciğer fonksiyonel rezervinin (Albumin-Bilirubin) objektif göstergesidir. Evrelendirmede kritik rol oynar.
2.  **APRI İndeksi:** SGOT (AST) ve Trombosit oranına bakarak karaciğer fibrozis (skarlaşma) derecesini cerrahi olmadan tahmin eder.
3.  **BAR (Bilirubin/Albumin Oranı):** Karaciğer hasarı (Bilirubin) ile sentez kapasitesi (Albumin) dengesini ölçer.
4.  **PAI (Protrombin / Albumin Endeksi):** Protrombin süresi (pıhtılaşma hızı) ile albümin sentez yeteneğini birleştirir.
5.  **Alk_Phos/SGOT Oranı:** Hasarın biliyer (safra yolu) kökenli mi yoksa hepatosellüler (hücre ölümü) kökenli mi olduğunu belirler.

---

## 🤖 Modelleme ve Açıklanabilirlik (Explainable AI)

*   **Model Seçimi:** `class_weight='balanced'` parametresi ve hiperparametre optimizasyonu yapılmış bir **LightGBM Classifier** final modeli olarak seçilmiştir.
*   **Küresel Açıklama (SHAP):** Bilirubin, Yaş, BAR, Protrombin ve Bakır değerlerinin tüm popülasyon genelinde tahmin üzerindeki global etkilerini gösterir.
*   **Yerel Açıklama (LIME):** Tekil hasta düzeyinde, karar sınırlarının arkasındaki sebepleri (örneğin bilurubinin normal olmasının hastayı hayatta tutma olasılığını nasıl artırdığını) açıklar.
*   **Kararlılık:** Model, 5-Fold Çapraz Doğrulamada son derece düşük standart sapma (%1.5) ile kararlı çalışmış ve genellenebilirliği kanıtlanmıştır.

---

## 📊 Streamlit Sunum Paneli

Proje, slayt veya sunum hazırlamaya gerek kalmaksızın, doğrudan Streamlit arayüzü üzerinden sunulabilecek bir **Sunum Paneli** haline getirilmiştir.

### 🧭 Arayüz Bölümleri ve Sunum Akışı:
1.  **Proje Özeti:** Siroz hastalığı durum tahmini projesinin amaçları, Mayo Clinic veri seti açıklaması ve C (Stabil), CL (Nakil), D (Vefat) sınıflarının klinik anlamları.
2.  **EDA Bulguları:** Sınıf dağılım grafikleri, eksik gözlem durumları, laboratuvar bulguları dağılımları, aykırı değer tespitleri ve klinik çıkarımlar (sekmeli olarak sunulmaktadır).
3.  **Ön İşleme & Özellik Mühendisliği:** Veri temizleme, dummy/one-hot encoding, status kodlaması ve üretilen yeni klinik skorların (ALBI, APRI, BAR, PAI, Alk_Phos/SGOT) LaTeX formülleri ve klinik açıklamaları.
4.  **Model Karşılaştırma:** LazyPredict algoritma tarama sonuç tablosu, en başarılı 3 modelin karşılaştırması ve neden LightGBM'in seçildiğine dair teknik gerekçeler.
5.  **Sınıf Ağırlıklandırma Deneyleri:** CL azınlık sınıfının recall skorunu artırmak amacıyla yapılan default vs. balanced LightGBM karşılaştırmaları, manuel ağırlık deneme tablosu, hata matrisleri ve hassasiyet/duyarlılık trade-off analizi.
6.  **Hiperparametre Optimizasyonu:** RandomizedSearchCV ve StratifiedKFold yaklaşımları, optimize edilmiş en iyi parametreler ve nihai test metrikleri.
7.  **Model Doğrulama (Cross-Validation):** 5-Fold Stratified Çapraz Doğrulama tablosu, ortalama doğruluk/F1 skorları, model Öğrenme Eğrisi (Learning Curve) ve aşırı öğrenme (overfitting) değerlendirmesi.
8.  **ROC Curve Analizi:** One-vs-Rest ROC eğrisi grafiği, sınıfların AUC değerleri (C: 0.9010, CL: 0.7908, D: 0.9047) ve sınıf ayrıştırma kabiliyetinin tartışılması.
9.  **SHAP & LIME Açıklanabilirlik:** Global özellik önem sıralaması (SHAP bar ve summary plots), tekil hasta düzeyinde kararların arkasındaki nedenler (LIME correct ve wrong predictions) ve küresel/yerel açıklama farkları.
10. **Canlı Tahmin Paneli:** Klinisyenin hasta verilerini sol menüden (sidebar) girerek siroz risk durumunu (Kompanse, Nakil adayı, Vefat riski) ve olasılık yüzdelerini anlık izlediği, hesaplanan klinik skorları ve klinik uyarıları gösteren etkileşimli ekran.
11. **Sonuç ve Sınırlılıklar:** Modelin başarı oranları, CL sınıfı veri kısıtı, akademik/prototip klinik karar destek sistemi yasal uyarıları ve gelecek çalışmaları içeren özet sunum kapanış slaytı.
