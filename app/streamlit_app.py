import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Siroz Tahmin ve Klinik Karar Destek Paneli",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Aesthetics and Layout Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 34px;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3d59 0%, #17b978 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 25px;
        text-align: center;
    }
    
    .section-title {
        font-size: 24px;
        font-weight: 600;
        color: #1e3d59;
        border-left: 5px solid #17b978;
        padding-left: 12px;
        margin-top: 25px;
        margin-bottom: 18px;
    }
    
    .clinical-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #eef1f6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .clinical-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }
    
    .status-card-c {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #1b5e20;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.08);
        font-size: 18px;
        font-weight: 600;
        border: 1px solid #a5d6a7;
    }
    
    .status-card-cl {
        background: linear-gradient(135deg, #fffde7 0%, #fff9c4 100%);
        color: #f57f17;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(245, 127, 23, 0.08);
        font-size: 18px;
        font-weight: 600;
        border: 1px solid #fff59d;
    }
    
    .status-card-d {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        color: #b71c1c;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(183, 28, 28, 0.08);
        font-size: 18px;
        font-weight: 600;
        border: 1px solid #ef9a9a;
    }
    
    .disclaimer-box {
        background-color: #fff9db;
        border-left: 6px solid #fab005;
        padding: 15px;
        border-radius: 4px;
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Find project root and define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'outputs', 'best_lgbm_model.pkl')

# Helper function to load model safely
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model(model_path)

# Helper function to load CSVs safely
def load_csv(filename):
    path = os.path.join(base_dir, 'outputs', filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Dosya okuma hatası ({filename}): {str(e)}")
            return None
    return None

# Helper function to display images safely
def display_image(path, caption):
    if os.path.exists(path):
        st.image(path, use_container_width=True, caption=caption)
    else:
        st.warning(f"⚠️ Görsel bulunamadı: {os.path.basename(path)} (Yol: {path})")

# Image paths definitions
lime_correct_path = os.path.join(base_dir, 'outputs', 'lime_correct_prediction.png')
lime_wrong_path = os.path.join(base_dir, 'outputs', 'lime_wrong_prediction.png')

# Sidebar Presentation Menu
st.sidebar.title("📌 Sunum Menüsü")
page = st.sidebar.radio(
    "Bölümler",
    [
        "1. Proje Özeti",
        "2. EDA Bulguları",
        "3. Ön İşleme & Özellik Mühendisliği",
        "4. Model Karşılaştırma",
        "5. Sınıf Ağırlıklandırma Deneyleri",
        "6. Hiperparametre Optimizasyonu",
        "7. Model Doğrulama (Cross-Validation)",
        "8. ROC Curve Analizi",
        "9. SHAP & LIME Açıklanabilirlik",
        "10. Canlı Tahmin Paneli",
        "11. Sonuç ve Sınırlılıklar"
    ]
)

# Render Patient Inputs ONLY for the "10. Canlı Tahmin Paneli"
if page == "10. Canlı Tahmin Paneli":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏥 Hasta Klinik Bilgileri")
    N_Days = st.sidebar.number_input("Takip Süresi (Gün)", min_value=1, max_value=5000, value=1800)
    Age_Years = st.sidebar.slider("Yaş (Yıl)", min_value=18, max_value=100, value=50)
    Drug = st.sidebar.selectbox("Kullanılan İlaç (Drug)", ["D-penicillamine", "Placebo"], index=1)
    Sex = st.sidebar.selectbox("Cinsiyet (Sex)", ["F", "M"], index=0)
    
    st.sidebar.markdown("### 📋 Klinik Belirtiler")
    Ascites = st.sidebar.selectbox("Asit Birikimi var mı? (Ascites)", ["N", "Y"], index=0)
    Hepatomegaly = st.sidebar.selectbox("Karaciğer Büyümesi var mı? (Hepatomegaly)", ["N", "Y"], index=0)
    Spiders = st.sidebar.selectbox("Örümcek Anjiyom var mı? (Spiders)", ["N", "Y"], index=0)
    Edema = st.sidebar.selectbox("Ödem Durumu (Edema)", ["N", "S", "Y"], index=0)
    Stage = st.sidebar.slider("Siroz Evresi (Stage)", min_value=1, max_value=4, value=3)
    
    st.sidebar.markdown("### 🧪 Laboratuvar Bulguları")
    Bilirubin = st.sidebar.number_input("Bilirubin (mg/dL)", min_value=0.1, max_value=30.0, value=1.2, step=0.1)
    Cholesterol = st.sidebar.number_input("Kolesterol (mg/dL)", min_value=100.0, max_value=2000.0, value=300.0, step=10.0)
    Albumin = st.sidebar.number_input("Albumin (g/dL)", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    Copper = st.sidebar.number_input("Bakır / Copper (ug/dL)", min_value=1.0, max_value=600.0, value=60.0, step=5.0)
    Alk_Phos = st.sidebar.number_input("Alkalen Fosfataz (U/L)", min_value=100.0, max_value=15000.0, value=1200.0, step=50.0)
    SGOT = st.sidebar.number_input("SGOT / AST (U/mL)", min_value=10.0, max_value=500.0, value=110.0, step=5.0)
    Tryglicerides = st.sidebar.number_input("Trigliserid (mg/dL)", min_value=10.0, max_value=600.0, value=110.0, step=5.0)
    Platelets = st.sidebar.number_input("Trombosit (k/mL)", min_value=10.0, max_value=600.0, value=265.0, step=10.0)
    Prothrombin = st.sidebar.number_input("Protrombin Zamanı (Saniye)", min_value=8.0, max_value=20.0, value=10.5, step=0.1)

    # Preprocessing & Feature Engineering Function
    def process_user_input():
        inputs = {
            'N_Days': N_Days,
            'Age': Age_Years * 365.25, # Years to days conversion
            'Bilirubin': Bilirubin,
            'Cholesterol': Cholesterol,
            'Albumin': Albumin,
            'Copper': Copper,
            'Alk_Phos': Alk_Phos,
            'SGOT': SGOT,
            'Tryglicerides': Tryglicerides,
            'Platelets': Platelets,
            'Prothrombin': Prothrombin,
            'Stage': Stage
        }
        df = pd.DataFrame([inputs])
        
        # One-hot encoding mapping (dropping drop_first=False equivalent logic)
        df['Drug_D-penicillamine'] = 1 if Drug == 'D-penicillamine' else 0
        df['Drug_Placebo'] = 1 if Drug == 'Placebo' else 0
        df['Sex_F'] = 1 if Sex == 'F' else 0
        df['Sex_M'] = 1 if Sex == 'M' else 0
        df['Ascites_N'] = 1 if Ascites == 'N' else 0
        df['Ascites_Y'] = 1 if Ascites == 'Y' else 0
        df['Hepatomegaly_N'] = 1 if Hepatomegaly == 'N' else 0
        df['Hepatomegaly_Y'] = 1 if Hepatomegaly == 'Y' else 0
        df['Spiders_N'] = 1 if Spiders == 'N' else 0
        df['Spiders_Y'] = 1 if Spiders == 'Y' else 0
        df['Edema_N'] = 1 if Edema == 'N' else 0
        df['Edema_S'] = 1 if Edema == 'S' else 0
        df['Edema_Y'] = 1 if Edema == 'Y' else 0
        
        # Feature Engineering Formulas
        df['ALBI'] = 0.66 * np.log10(df['Bilirubin'] * 17.1) - 0.085 * (df['Albumin'] * 10)
        df['APRI'] = ((df['SGOT'] / 40.0) * 100) / df['Platelets']
        df['BAR'] = df['Bilirubin'] / df['Albumin']
        df['PAI'] = df['Prothrombin'] / df['Albumin']
        df['Alk_Phos_SGOT_Ratio'] = df['Alk_Phos'] / df['SGOT']
        
        # Birebir processed_train.csv sütun sırası (Status hariç)
        feature_cols = [
            'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage',
            'Drug_D-penicillamine', 'Drug_Placebo', 'Sex_F', 'Sex_M', 'Ascites_N', 'Ascites_Y', 'Hepatomegaly_N', 'Hepatomegaly_Y', 'Spiders_N', 'Spiders_Y', 'Edema_N', 'Edema_S', 'Edema_Y',
            'ALBI', 'APRI', 'BAR', 'PAI', 'Alk_Phos_SGOT_Ratio'
        ]
        return df[feature_cols]

# Header
st.markdown("<div class='main-title'>🏥 Karaciğer Sirozu Klinik Sunum ve Karar Destek Paneli</div>", unsafe_allow_html=True)

# Academic disclaimer shown globally in small fonts at the top
st.markdown("""
<div class='disclaimer-box'>
    ⚠️ AKADEMİK / PROTOTİP KARAR DESTEK UYARISI: Bu uygulama klinik araştırmalar ve model performansı sunumları için geliştirilmiş bir prototiptir. Tahminler kesin bir tıbbi teşhis niteliği taşımaz ve hekim kararlarının yerine geçemez.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROJE ÖZETİ
# ─────────────────────────────────────────────────────────────────────────────
if page == "1. Proje Özeti":
    st.markdown("<div class='section-title'>1. Proje Özeti</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='clinical-card'>
            <h3>🎯 Projenin Amacı</h3>
            <p>Bu proje, Karaciğer Sirozu (özellikle Primer Biliyer Siroz - PBC) hastalarının demografik, klinik semptom ve laboratuvar tahlil bulgularını değerlendirerek, hastaların durumunu tahmin etmeyi amaçlar. Sistem, klinisyenlere tıbbi kararlarında rehberlik edebilecek bir yapay zeka karar destek aracıdır.</p>
        </div>
        <div class='clinical-card'>
            <h3>📂 Veri Seti Açıklaması</h3>
            <p>Kullanılan veri seti, <b>Mayo Clinic</b> tarafından 1974-1984 yılları arasında gerçekleştirilen primer biliyer siroz klinik denemesine dayanmaktadır. Veri setinde hastaların yaş, cinsiyet gibi demografik bilgilerinin yanında, bilirubin, albümin, kolesterol ve pıhtılaşma süreleri gibi kritik karaciğer rezerv göstergeleri yer almaktadır.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='clinical-card'>
            <h3>🏷️ Hedef Değişken: Status</h3>
            <p>Hastaların klinik takipleri sonucundaki durumunu gösteren <b>Status</b> değişkeni modelimizin tahmin hedefidir:</p>
            <ul>
                <li><span class='badge-c'>C</span> <b>Kompanse / Stabil (Sınıf 0):</b> Hasta hayattadır, karaciğer fonksiyonları dengededir. Takip ve konservatif tedavi önerilir.</li>
                <li><span class='badge-cl'>CL</span> <b>Nakil Gerekli / Yaşıyor (Sınıf 1):</b> Hastaya karaciğer nakli yapılmıştır veya acil nakil adayıdır.</li>
                <li><span class='badge-d'>D</span> <b>Vefat Riski / Ağır Durum (Sınıf 2):</b> Siroz komplikasyonları nedeniyle vefat durumu gerçekleşmiştir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA BULGULARI
# ─────────────────────────────────────────────────────────────────────────────
elif page == "2. EDA Bulguları":
    st.markdown("<div class='section-title'>2. Keşifçi Veri Analizi (EDA) Bulguları</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Sınıf Dağılımı & Eksik Değer Durumu", 
        "📈 Sayısal Değişken Dağılımları & Aykırı Değerler", 
        "🔗 Korelasyon Heatmap"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            img_target = os.path.join(base_dir, 'reports', 'figures', 'target_distribution.png')
            display_image(img_target, "Sınıf Dağılımı")
        with col2:
            img_missing = os.path.join(base_dir, 'reports', 'figures', 'missing_values.png')
            display_image(img_missing, "Eksik Değer Durumu")
        
        st.markdown("""
        <div class='clinical-card'>
            <b>📊 Sınıf Dağılımı Yorumu:</b><br>
            Veri setinde sınıflar arasında ciddi bir dengesizlik (imbalance) vardır. 
            Transplant yapılan hastaları temsil eden <b>CL</b> sınıfı toplam verinin %3'ünden daha azını oluşturmaktadır. 
            Bu durum, makine öğrenmesi algoritmalarının CL sınıfını öğrenmesini zorlaştıran temel etkendir. 
            Eksik değer analizi grafiğine göre veri setindeki eksik gözlemler temizlenmiş veya impute edilmiştir.
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            img_dist = os.path.join(base_dir, 'reports', 'figures', 'numerical_distributions.png')
            display_image(img_dist, "Sayısal Değişkenlerin Dağılımları")
        with col2:
            img_box = os.path.join(base_dir, 'reports', 'figures', 'numerical_boxplots.png')
            display_image(img_box, "Aykırı Değer (Outlier) Bulguları")
            
        st.markdown("""
        <div class='clinical-card'>
            <b>📈 Dağılım ve Aykırı Değer Yorumu:</b><br>
            Bilirubin, Bakır (Copper), Alkalen Fosfataz (Alk_Phos), Trigliserid (Tryglicerides) gibi laboratuvar değişkenlerinin dağılımları oldukça sağa çarpıktır (positively skewed). 
            Klinik olarak bu sağa çarpıklık ve boxplot'larda görülen aşırı yüksek aykırı değerler sirozun ciddiyetiyle ve karaciğer hasarının derecesiyle doğrudan örtüşmektedir. 
            Bu aykırı değerler veri hatası değil, gerçek hasta durumlarını yansıttığı için temizlenmek yerine modele dahil edilmiştir.
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        img_corr = os.path.join(base_dir, 'reports', 'figures', 'correlation_matrix.png')
        display_image(img_corr, "Korelasyon Heatmap")
        
        st.markdown("""
        <div class='clinical-card'>
            <b>🔗 Korelasyon Yorumu:</b><br>
            Korelasyon matrisinde, Bilirubin ve Prothrombin zamanının (pıhtılaşma süresi) hastalık evresiyle pozitif yönde ilişkili olduğu görülürken, Albumin protein sentez yeteneğini yansıttığı için negatif yönde korelasyon sergilemektedir. 
            Ayrıca safra yolu hasarını yansıtan Alk_Phos ile hepatosellüler hasarı yansıtan SGOT enzimleri arasında anlamlı tıbbi ilişkiler izlenmektedir.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "3. Ön İşleme & Özellik Mühendisliği":
    st.markdown("<div class='section-title'>3. Ön İşleme & Özellik Mühendisliği</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("""
        <div class='clinical-card'>
            <h3>⚙️ Ön İşleme Adımları</h3>
            <ul>
                <li><b>id Sütununun Kaldırılması:</b> Model için bilgi taşımayan ve ezberlemeye sebep olan id sütunu veri setinden atılmıştır.</li>
                <li><b>Status Kodlama:</b> Hedef sınıf etiketleri tamsayıya kodlanmıştır: <code>C: 0, CL: 1, D: 2</code>.</li>
                <li><b>One-Hot Encoding:</b> Drug, Sex, Ascites, Hepatomegaly, Spiders, Edema gibi kategorik değişkenler kukla (dummy) değişkenlere dönüştürülmüştür.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### 🧬 Yeni Klinik Skorlar (Feature Engineering)")
        
        st.markdown("**1. ALBI (Albumin-Bilirubin) Skoru:**")
        st.latex(r"ALBI = 0.66 \times \log_{10}(\text{Bilirubin} \times 17.1) - 0.085 \times (\text{Albumin} \times 10)")
        st.markdown("""
        * *Klinik Açıklama:* Karaciğerin sentez kapasitesi (Albumin) ve temizleme gücünü (Bilirubin) birleştiren, literatürde siroz evrelemesinde kullanılan objektif bir skordur.
        """)
        
        st.markdown("**2. APRI (AST to Platelet Ratio Index):**")
        st.latex(r"APRI = \frac{(\text{SGOT} / 40.0) \times 100}{\text{Platelets}}")
        st.markdown("""
        * *Klinik Açıklama:* SGOT (AST) enzim yüksekliği ile trombosit (Platelet) düşüşünü oranlayarak karaciğer fibrozis/siroz skarlaşma derecesini biyopsisiz tahmin etmeye yarar.
        """)
        
        st.markdown("**3. BAR (Bilirubin / Albumin Oranı):**")
        st.latex(r"BAR = \frac{\text{Bilirubin}}{\text{Albumin}}")
        st.markdown("""
        * *Klinik Açıklama:* Karaciğer hasarı (Bilirubin) ve sentez gücü (Albumin) dengesini yansıtan basit ve hassas bir orandır.
        """)
        
        st.markdown("**4. PAI (Protrombin / Albumin Endeksi):**")
        st.latex(r"PAI = \frac{\text{Prothrombin}}{\text{Albumin}}")
        st.markdown("""
        * *Klinik Açıklama:* Pıhtılaşma proteini sentez yeteneği (Protrombin süresi) ile albümin sentez gücünü birleştiren sentez rezerv göstergesidir.
        """)
        
        st.markdown("**5. Alk_Phos / SGOT Oranı:**")
        st.latex(r"\text{Oran} = \frac{\text{Alk\_Phos}}{\text{SGOT}}")
        st.markdown("""
        * *Klinik Açıklama:* Sirozun biliyer/safra kanalı hasarı (Alk_Phos) kökenli mi yoksa aktif hepatosellüler hasar/hücre ölümü (SGOT) kökenli mi olduğunu ayırt eder.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL KARŞILAŞTIRMA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "4. Model Karşılaştırma":
    st.markdown("<div class='section-title'>4. Model Karşılaştırma ve LightGBM Seçimi</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("#### LazyPredict Sonuçları (İlk 10 Model)")
        lp_df = load_csv('lazypredict_results.csv')
        if lp_df is not None:
            st.dataframe(lp_df.head(10), use_container_width=True)
        else:
            st.warning("LazyPredict sonuç tablosu bulunamadı.")
            
        st.markdown("""
        <div class='clinical-card'>
            <h3>🏥 En İyi 3 Model Karşılaştırması</h3>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='background-color:#f8f9fa;'>
                    <th style='padding:8px; border:1px solid #ddd;'>Model</th>
                    <th style='padding:8px; border:1px solid #ddd;'>Genel Doğruluk</th>
                    <th style='padding:8px; border:1px solid #ddd;'>Macro F1</th>
                </tr>
                <tr>
                    <td style='padding:8px; border:1px solid #ddd;'><b>LightGBM Classifier</b></td>
                    <td style='padding:8px; border:1px solid #ddd;'>0.8286</td>
                    <td style='padding:8px; border:1px solid #ddd;'><b>0.6378</b></td>
                </tr>
                <tr>
                    <td style='padding:8px; border:1px solid #ddd;'>RandomForest Classifier</td>
                    <td style='padding:8px; border:1px solid #ddd;'>0.8185</td>
                    <td style='padding:8px; border:1px solid #ddd;'>0.5660</td>
                </tr>
                <tr>
                    <td style='padding:8px; border:1px solid #ddd;'>Bagging Classifier</td>
                    <td style='padding:8px; border:1px solid #ddd;'>0.7951</td>
                    <td style='padding:8px; border:1px solid #ddd;'>0.5562</td>
                </tr>
            </table>
        </div>
        <div class='clinical-card'>
            <b>Neden LightGBM?</b><br>
            LightGBM, LazyPredict taramasında en yüksek Macro F1 değerini vermiştir. 
            Ayrıca yüksek eğitim hızı, sınıf ağırlıklandırma parametresi (<code>class_weight</code>) desteği ve ağaç tabanlı yapı sayesinde klinik verilerdeki çarpıklıklara karşı kararlı duruşu nedeniyle tercih edilmiştir.
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        img_comp = os.path.join(base_dir, 'outputs', 'model_comparison.png')
        display_image(img_comp, "LazyPredict Algoritmaları Karşılaştırma Grafiği")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SINIF AĞIRLIKLANDIRMA DENEYLERİ
# ─────────────────────────────────────────────────────────────────────────────
elif page == "5. Sınıf Ağırlıklandırma Deneyleri":
    st.markdown("<div class='section-title'>5. Sınıf Ağırlıklandırma (Class Weight) ve CL Sınıfı Analizi</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("#### Varsayılan LightGBM vs Dengeli (Balanced) LightGBM")
        lgbm_cw_df = load_csv('lgbm_class_weight_results.csv')
        if lgbm_cw_df is not None:
            st.dataframe(lgbm_cw_df, use_container_width=True)
        else:
            st.warning("Varsayılan vs Balanced LightGBM sonuç tablosu bulunamadı.")
            
        st.markdown("#### Farklı Class Weight Kombinasyonlarının Deney Sonuçları")
        cw_df = load_csv('class_weight_experiment.csv')
        if cw_df is not None:
            st.dataframe(cw_df, use_container_width=True)
        else:
            st.warning("Manuel class weight deney sonuç tablosu bulunamadı.")
            
        st.markdown("""
        <div class='clinical-card'>
            <b>⚖️ CL Sınıfı Recall / F1 Yorumu:</b><br>
            Varsayılan modelde CL recall değeri %14.54 seviyesindeyken, sınıf dengesizliğini gidermek için <code>class_weight='balanced'</code> parametresi kullanıldığında bu oran <b>%27.27</b>'ye yükselmiştir.<br>
            <code>balanced</code> ayarı, CL sınıfı için otomatik olarak yaklaşık <b>18 kat</b> daha yüksek ağırlık tanımlayarak modelin bu azınlık sınıfı örneklerini kaçırmamasını sağlar. 
            Manuel ağırlıklarda (1:10:1 veya 1:15:1) precision hafifçe artsa da, klinik hedefin "nakil ihtiyacı olan hastaları gözden kaçırmamak" olması nedeniyle, en yüksek CL Recall veren <b>balanced LightGBM</b> modeli final adayı olarak seçilmiştir.
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        img_cw = os.path.join(base_dir, 'outputs', 'class_weight_comparison.png')
        display_image(img_cw, "Sınıf Ağırlıklandırma Metotları Hata Matrisi (Confusion Matrix) Karşılaştırması")

# ─────────────────────────────────────────────────────────────────────────────
# 6. HİPERPARAMETRE OPTİMİZASYONU
# ─────────────────────────────────────────────────────────────────────────────
elif page == "6. Hiperparametre Optimizasyonu":
    st.markdown("<div class='section-title'>6. Hiperparametre Optimizasyonu</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class='clinical-card'>
            <h3>⚙️ Optimizasyon Metodolojisi</h3>
            <p><b>RandomizedSearchCV:</b> Hiperparametre uzayından rassal kombinasyonlar seçerek 50 iterasyon boyunca en iyi çapraz doğrulama Macro F1 skoruna sahip hiperparametreleri aramıştır.</p>
            <p><b>StratifiedKFold (5-Fold):</b> Sınıf dengesizliği yüksek olduğu için fold bölmelerinde orijinal sınıf dağılımı korunmuş, modelin doğrulamada adil değerlendirilmesi sağlanmıştır.</p>
        </div>
        """, unsafe_allow_html=True)
        
        hp_df = load_csv('hyperparameter_results.csv')
        if hp_df is not None:
            st.markdown("#### Hiperparametre Arama Sonuç Özeti")
            st.write(hp_df)
        else:
            st.warning("Hiperparametre sonuç tablosu bulunamadı.")
            
    with col2:
        if hp_df is not None:
            best_params = hp_df.iloc[0]['Best Parameters']
            st.markdown("#### 🎯 En İyi Hiperparametreler")
            st.code(best_params, language='json')
            
            st.markdown("""
            <div class='clinical-card'>
                <b>Optimizasyon Yorumu:</b><br>
                Modelin karmaşıklığını ve ezberlemesini sınırlamak için <code>max_depth=5</code> ve yüksek <code>reg_alpha=5.0</code> değerleri seçilmiştir. 
                Bu durum eğitim setinde aşırı öğrenmeyi (overfitting) engellerken, test seti F1 skorunu kararlı hale getirmiştir.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL DOĞRULAMA (CROSS-VALIDATION)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "7. Model Doğrulama (Cross-Validation)":
    st.markdown("<div class='section-title'>7. Model Doğrulama & Öğrenme Eğrisi</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 5-Fold Cross-Validation Sonuç Tablosu")
        cv_df = load_csv('cross_validation_results.csv')
        if cv_df is not None:
            st.dataframe(cv_df, use_container_width=True)
            
            # Extract Mean metrics
            mean_row = cv_df[cv_df['Fold'] == 'Mean']
            if not mean_row.empty:
                st.markdown("#### 📊 Ortalama Doğrulama Metrikleri (Mean ± Std)")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Val Accuracy", "81.62% ± 0.45%")
                with col_m2:
                    st.metric("Val Macro F1", "66.57% ± 1.59%")
        else:
            st.warning("Cross validation sonuç tablosu bulunamadı.")
            
        st.markdown("""
        <div class='clinical-card'>
            <b>📈 Overfitting (Aşırı Öğrenme) ve Öğrenme Eğrisi Analizi:</b><br>
            Eğitim F1 skoru (%91.33) ile doğrulama F1 skoru (%66.57) arasında bir fark (gap) bulunsa da, fold'lar arasındaki standart sapmanın son derece düşük olması (%1.59) modelin kararlılığını gösterir.<br>
            Learning Curve grafiği incelendiğinde, veri boyutu arttıkça doğrulama F1 skorunun yukarı doğru ivmelendiği ve eğitim skorunun dengelenmeye başladığı görülmektedir. 
            Bu durum, veri boyutunun artırılmasının modelin genel performansını daha da iyileştireceğine işaret eder.
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        img_lc = os.path.join(base_dir, 'outputs', 'learning_curve.png')
        display_image(img_lc, "Model Öğrenme Eğrisi (Learning Curve)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ROC CURVE ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────
elif page == "8. ROC Curve Analizi":
    st.markdown("<div class='section-title'>8. Sınıf Ayrıştırma Yeteneği (ROC Curve)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        img_roc = os.path.join(base_dir, 'outputs', 'roc_curve_ovr.png')
        display_image(img_roc, "One-vs-Rest (OvR) ROC Eğrisi")
        
    with col2:
        st.markdown("#### Sınıf Bazlı Alan Değerleri (AUC)")
        roc_df = load_csv('roc_results.csv')
        if roc_df is not None:
            st.table(roc_df)
        else:
            st.warning("ROC sonuç tablosu bulunamadı.")
            
        st.markdown("""
        <div class='clinical-card'>
            <b>📊 ROC & AUC Yorumu:</b><br>
            Modelin genel <b>Macro Average AUC skoru 0.8655</b>'tir. Sınıf bazında incelendiğinde:
            <ul>
                <li><b>C (Stabil):</b> AUC = <b>0.9010</b> (Çok yüksek ayrıştırma gücü)</li>
                <li><b>D (Vefat):</b> AUC = <b>0.9047</b> (Çok yüksek ayrıştırma gücü)</li>
                <li><b>CL (Nakil):</b> AUC = <b>0.7908</b> (Düşük veri adedine rağmen başarılı ayrıştırma)</li>
            </ul>
            AUC değerlerinin 0.80-0.90 bandı ve üzerinde olması, karar destek sisteminin olasılık eşiklerinin ayrıştırma kalitesinin mükemmel olduğunu tesciller.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP & LIME AÇIKLANABİLİRLİK
# ─────────────────────────────────────────────────────────────────────────────
elif page == "9. SHAP & LIME Açıklanabilirlik":
    st.markdown("<div class='section-title'>9. Küresel ve Yerel Açıklanabilirlik (SHAP & LIME)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='clinical-card'>
        <h3>🧠 Küresel (Global) vs Yerel (Local) Açıklama</h3>
        <p><b>SHAP:</b> Modelin genel olarak tüm veri kümesindeki davranışını açıklar. Özelliklerin model kararı üzerindeki kooperatif önemini gösterir.</p>
        <p><b>LIME:</b> Tekil bir hastanın tahmini arkasındaki karar sınırlarını açıklar. O hasta için hangi lab sonucunun riski artırdığını veya düşürdüğünü gösterir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🌎 SHAP Global Analizi", "📍 LIME Yerel Hasta Örnekleri"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.markdown("#### SHAP En Önemli Değişkenler Tablosu")
            shap_df = load_csv('shap_importance.csv')
            if shap_df is not None:
                st.dataframe(shap_df.head(10).round(4), use_container_width=True)
            else:
                st.warning("SHAP önem tablosu bulunamadı.")
        with col2:
            img_shap_bar = os.path.join(base_dir, 'outputs', 'shap_bar.png')
            display_image(img_shap_bar, "Ortalama Mutlak SHAP Değerleri")
            
        img_shap_sum = os.path.join(base_dir, 'outputs', 'shap_summary.png')
        display_image(img_shap_sum, "SHAP Summary Plot")
        
        st.markdown("""
        <div class='clinical-card'>
            <b>📝 SHAP Klinik Önem Çıkarımı:</b><br>
            Modelin en önemli özellikleri sırasıyla <b>N_Days</b> (Takip süresi), <b>Age</b> (Yaş), <b>BAR</b> (Bilirubin/Albumin Oranı) ve <b>Prothrombin</b> (Protrombin zamanı)'dir.<br>
            Bilirubinin artışı ve albuminin düşmesi (BAR oranının yükselmesi) vefat yönündeki (D) riski doğrudan artırırken; daha genç yaşta olmak ve stabil albümin düzeyleri hastanın hayatta kalma olasılığını (C) tetiklemektedir. 
            Feature engineering ile ürettiğimiz BAR oranının global önem sıralamasında 3. sırada yer alması, klinik özellik mühendisliğinin modele katkısını kanıtlamaktadır.
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            display_image(lime_correct_path, "Modelin Doğru Tahmin Ettiği Hasta (LIME)")
        with col2:
            display_image(lime_wrong_path, "Modelin Yanlış Tahmin Ettiği Hasta (LIME)")
            
        st.markdown("""
        <div class='clinical-card'>
            <b>📍 LIME Grafikleri Analizi:</b><br>
            Doğru tahmin edilen hastada, bilirubin seviyesinin düşük olması ve stabil albümin düzeyi, modeli hastanın C (Hayatta) durumunda olduğuna ikna etmiştir.<br>
            Yanlış tahmin edilen hastada ise, normal dışı laboratuvar bulguları ve yaş faktörü vefat olasılığını desteklemesine rağmen, hastanın gerçekte kompanse siroz durumunda olması, sınır değerlerdeki hastaların klinik karar desteğinde hekim kontrolünün önemini ortaya koymaktadır.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. CANLI TAHMİN PANELİ
# ─────────────────────────────────────────────────────────────────────────────
elif page == "10. Canlı Tahmin Paneli":
    st.markdown("<div class='section-title'>10. Canlı Tahmin Paneli (Karar Destek)</div>", unsafe_allow_html=True)
    
    if model is not None:
        processed_df = process_user_input()
        
        # Predict probabilities
        proba = model.predict_proba(processed_df)[0]
        pred_class_idx = np.argmax(proba)
        confidence = proba[pred_class_idx]
        
        class_labels = {0: 'C', 1: 'CL', 2: 'D'}
        class_names_tr = {
            'C': 'Kompanse / Stabil (Hayatta)',
            'CL': 'Karaciğer Nakli Gerekli (Yaşıyor)',
            'D': 'Vefat Riski / Ağır Durum'
        }
        
        predicted_label = class_labels[pred_class_idx]
        predicted_name_tr = class_names_tr[predicted_label]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 🎯 Tahmin Çıktısı")
            if predicted_label == 'C':
                st.markdown(f"<div class='status-card-c'>Durum: {predicted_name_tr}<br>Güven Skoru (Olasılık): %{confidence*100:.2f}</div>", unsafe_allow_html=True)
                st.markdown("💡 **Klinik Değerlendirme Önerisi:** Hastanın karaciğer sentez ve temizleme kapasitesi büyük oranda korunmuştur. Hekim kontrolünde rutin poliklinik takibi ve konservatif tedavi sürdürülebilir.")
            elif predicted_label == 'CL':
                st.markdown(f"<div class='status-card-cl'>Durum: {predicted_name_tr}<br>Güven Skoru (Olasılık): %{confidence*100:.2f}</div>", unsafe_allow_html=True)
                st.markdown("⚠️ **Klinik Değerlendirme Önerisi:** Karaciğer rezervlerinde ciddi azalma ve nakil ihtiyacı belirtileri mevcuttur. Hastanın acilen organ nakli (transplantasyon) kurullarınca değerlendirilmesi önerilir.")
            else:
                st.markdown(f"<div class='status-card-d'>Durum: {predicted_name_tr}<br>Güven Skoru (Olasılık): %{confidence*100:.2f}</div>", unsafe_allow_html=True)
                st.markdown("🚨 **Klinik Değerlendirme Önerisi:** Siroza bağlı mortalite/vefat riski yüksektir. Hastanın yatırılarak takip edilmesi ve siroz komplikasyonlarına (özefagus varis kanaması, hepatik ensefalopati, asit peritoniti vb.) yönelik acil tedavilerin planlanması gerekebilir.")
                
        with col2:
            st.markdown("#### 📊 Sınıf Olasılıkları Dağılımı")
            proba_df = pd.DataFrame({
                'Sınıf': ['Stabil (C)', 'Nakil Adayı (CL)', 'Ağır Durum (D)'],
                'Olasılık (%)': proba * 100
            })
            
            fig, ax = plt.subplots(figsize=(6, 3.8))
            colors = [
                '#2e7d32' if predicted_label == 'C' else '#a5d6a7',
                '#f57f17' if predicted_label == 'CL' else '#fff59d',
                '#c62828' if predicted_label == 'D' else '#ef9a9a'
            ]
            
            sns.barplot(x='Olasılık (%)', y='Sınıf', data=proba_df, palette=colors, ax=ax)
            ax.set_xlim(0, 100)
            ax.set_title("Durum Olasılık Yüzdeleri", fontweight='bold', fontsize=12, color='#1e3d59')
            for index, value in enumerate(proba_df['Olasılık (%)']):
                ax.text(value + 1, index, f"%{value:.1f}", va='center', fontweight='bold', fontsize=10)
            st.pyplot(fig)
            plt.close()
            
        # Display Calculated Clinical Scores
        st.markdown("<div class='section-title'>Hesaplanan Klinik Skorlar</div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        
        # ALBI Category Calculation
        albi_val = processed_df['ALBI'].values[0]
        if albi_val < -2.60:
            albi_grade = "Grade 1 (İyi Karaciğer)"
        elif albi_val <= -1.39:
            albi_grade = "Grade 2 (Orta Karaciğer)"
        else:
            albi_grade = "Grade 3 (Zayıf Karaciğer)"
            
        with col_s1:
            st.metric("ALBI Skoru", f"{albi_val:.3f}")
            st.caption(f"Evre: {albi_grade}")
        with col_s2:
            st.metric("APRI Skoru", f"{processed_df['APRI'].values[0]:.3f}")
            st.caption("Fibrozis Göstergesi")
        with col_s3:
            st.metric("BAR (Bilirubin/Albumin)", f"{processed_df['BAR'].values[0]:.3f}")
            st.caption("Hasar/Sentez Oranı")
        with col_s4:
            st.metric("PAI (Protrombin/Albumin)", f"{processed_df['PAI'].values[0]:.3f}")
            st.caption("Sentez Kapasitesi")
        with col_s5:
            st.metric("Alk_Phos / SGOT Oranı", f"{processed_df['Alk_Phos_SGOT_Ratio'].values[0]:.3f}")
            st.caption("Hasar Kökeni Göstergesi")
            
        st.markdown("""
        <div class='disclaimer-box' style='margin-top:20px;'>
            ⚠️ KLİNİK YASAL UYARI: Yukarıdaki metrik hesaplamaları ve olasılık dağılımları, LightGBM makine öğrenmesi modelinin veri seti örüntülerine dayanarak ürettiği tahminlerdir. Bu bilgiler herhangi bir klinik tanı, tıbbi reçete veya resmi tedavi planı oluşturmaz. Kararlar yalnızca yetkili hekimler tarafından verilmelidir.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Tahmin modeli best_lgbm_model.pkl yüklenemedi. Tahmin gerçekleştirilemiyor.")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SONUÇ VE SINIRLILIKLAR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "11. Sonuç ve Sınırlılıklar":
    st.markdown("<div class='section-title'>11. Sonuç ve Sınırlılıklar</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='clinical-card'>
        <h3>📝 Proje Sonuçları</h3>
        <ul>
            <li><b>Sınıf Ayrıştırma Başarısı:</b> Model, veri setinde yoğunlukları yüksek olan stabil/hayatta (C) ve vefat (D) durumlarındaki hastaları %90'ın üzerinde yüksek AUC başarı oranlarıyla ayırt edebilmektedir.</li>
            <li><b>Class Weight Parametresi:</b> Az örnek sayılı CL (transplant) sınıfının yakalanma oranı (Recall), <code>class_weight='balanced'</code> stratejisiyle belirgin derecede iyileştirilmiştir. Bu durum, acil nakil ihtiyacı olan kritik vakaların atlanmasının önlenmesi açısından tıbbi fayda sağlamaktadır.</li>
            <li><b>Explainable AI (Açıklanabilirlik):</b> SHAP ve LIME yöntemleriyle model kararlarının arkasındaki lab tahlil ağırlıkları şeffaflaştırılmış, karar sınırları hekimlerin doğrulayabileceği biçimde görselleştirilmiştir.</li>
        </ul>
    </div>
    <div class='clinical-card'>
        <h3>⚠️ Klinik Kısıtlar ve Uyarılar</h3>
        <ul>
            <li><b>Klinik Teşhis Yerine Geçmez:</b> Bu panel bir akademik/prototip klinik karar destek aracıdır. Üretilen çıktılar resmi bir hekim tanısı niteliği taşımaz.</li>
            <li><b>Düşük CL Örnek Sayısı:</b> Karaciğer nakli yapılan hastaların sayısı veride çok az olduğundan, model bu sınıfın özelliklerini ve karar sınırlarını öğrenmekte zorlanmaktadır.</li>
            <li><b>Veri Geliştirme İhtiyacı:</b> Modelin doğruluğunu ve genellenebilirliğini artırmak için gelecekte daha fazla sayıda ve özellikle dengeli hasta verisiyle çalışılması önerilmektedir.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
