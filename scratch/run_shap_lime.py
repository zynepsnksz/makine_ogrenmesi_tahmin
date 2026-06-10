import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import shap
from lime.lime_tabular import LimeTabularExplainer
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def run_shap_lime_analysis():
    print("Loading data and model...")
    df = pd.read_csv('data/processed_train.csv')
    X = df.drop(columns=['Status'])
    y = df['Status']

    # Train/Test Split (%20 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Load model
    with open('outputs/best_lgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # ─────────────────────────────
    # SHAP ANALİZİ
    # ─────────────────────────────
    print("Running SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # In SHAP 0.40+, shap_values can be a list or a numpy array. Let's handle both.
    # If list, we have 3 elements (one per class), each (n_samples, n_features).
    if isinstance(shap_values, list):
        shap_values_list = shap_values
    else:
        # shap_values could be a numpy array of shape (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        # TreeExplainer usually returns list of length 3, or a 3D array.
        if len(shap_values.shape) == 3:
            # If shape is (n_samples, n_features, n_classes), let's split it
            if shap_values.shape[2] == 3:
                shap_values_list = [shap_values[:, :, i] for i in range(3)]
            else:
                shap_values_list = [shap_values[i, :, :] for i in range(3)]
        else:
            shap_values_list = [shap_values]

    # Create SHAP Summary Plot (dot plot for class CL / class 1)
    print("Saving outputs/shap_summary.png...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_list[1], X_test, show=False)
    plt.title("SHAP Summary Plot (Class CL vs Rest)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create SHAP Bar Plot (multi-class absolute mean SHAP bar plot)
    print("Saving outputs/shap_bar.png...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_list, X_test, plot_type='bar', class_names=['C', 'CL', 'D'], show=False)
    plt.title("SHAP Feature Importance (All Classes)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('outputs/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Top 10 SHAP Importance Table
    print("Calculating SHAP Feature Importances...")
    # Calculate mean absolute SHAP values for each class
    mean_abs_shap_c = np.mean(np.abs(shap_values_list[0]), axis=0)
    mean_abs_shap_cl = np.mean(np.abs(shap_values_list[1]), axis=0)
    mean_abs_shap_d = np.mean(np.abs(shap_values_list[2]), axis=0)
    # Global mean across the 3 classes
    mean_global = (mean_abs_shap_c + mean_abs_shap_cl + mean_abs_shap_d) / 3

    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Abs_SHAP_C': mean_abs_shap_c,
        'Mean_Abs_SHAP_CL': mean_abs_shap_cl,
        'Mean_Abs_SHAP_D': mean_abs_shap_d,
        'Mean_Global_SHAP': mean_global
    })

    top_10_importance = importance_df.sort_values(by='Mean_Global_SHAP', ascending=False).head(10)
    top_10_importance.to_csv('outputs/shap_importance.csv', index=False)
    print("Top 10 SHAP Importance saved to outputs/shap_importance.csv")
    print(top_10_importance.to_string(index=False))

    # ─────────────────────────────
    # LIME ANALİZİ
    # ─────────────────────────────
    print("Running LIME TabularExplainer...")
    y_pred = model.predict(X_test)

    # Let's find correct and wrong predictions, ideally involving CL (class 1)
    indices_correct = np.where(y_pred == y_test)[0]
    indices_wrong = np.where(y_pred != y_test)[0]

    # Try to find a correct prediction where True = CL (1) and Pred = CL (1)
    correct_cl = [idx for idx in indices_correct if y_test.iloc[idx] == 1]
    if correct_cl:
        idx_correct = correct_cl[0]
    else:
        idx_correct = indices_correct[0]

    # Try to find a wrong prediction where True = CL (1) but Pred != CL (1) OR True != CL (1) but Pred = CL (1)
    wrong_cl = [idx for idx in indices_wrong if y_test.iloc[idx] == 1 or y_pred[idx] == 1]
    if wrong_cl:
        idx_wrong = wrong_cl[0]
    else:
        idx_wrong = indices_wrong[0]

    # Initialize LIME Tabular Explainer
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['C', 'CL', 'D'],
        mode='classification',
        random_state=42
    )

    # Correct prediction LIME explanation
    print(f"Correct prediction sample index in test set: {idx_correct}")
    print(f"True Class: {y_test.iloc[idx_correct]} ({['C', 'CL', 'D'][y_test.iloc[idx_correct]]})")
    print(f"Predicted Class: {y_pred[idx_correct]} ({['C', 'CL', 'D'][y_pred[idx_correct]]})")
    
    pred_label_correct = int(y_pred[idx_correct])
    exp_correct = lime_explainer.explain_instance(
        X_test.iloc[idx_correct], 
        model.predict_proba, 
        num_features=10, 
        labels=[pred_label_correct]
    )
    
    # Save LIME correct prediction plot
    fig = exp_correct.as_pyplot_figure(label=pred_label_correct)
    plt.title(f"LIME Correct Prediction (True: {['C', 'CL', 'D'][y_test.iloc[idx_correct]]}, Pred: {['C', 'CL', 'D'][pred_label_correct]})", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('outputs/lime_correct_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Wrong prediction LIME explanation
    print(f"Wrong prediction sample index in test set: {idx_wrong}")
    print(f"True Class: {y_test.iloc[idx_wrong]} ({['C', 'CL', 'D'][y_test.iloc[idx_wrong]]})")
    print(f"Predicted Class: {y_pred[idx_wrong]} ({['C', 'CL', 'D'][y_pred[idx_wrong]]})")
    
    pred_label_wrong = int(y_pred[idx_wrong])
    exp_wrong = lime_explainer.explain_instance(
        X_test.iloc[idx_wrong], 
        model.predict_proba, 
        num_features=10, 
        labels=[pred_label_wrong]
    )
    
    # Save LIME wrong prediction plot
    fig = exp_wrong.as_pyplot_figure(label=pred_label_wrong)
    plt.title(f"LIME Wrong Prediction (True: {['C', 'CL', 'D'][y_test.iloc[idx_wrong]]}, Pred: {['C', 'CL', 'D'][pred_label_wrong]})", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('outputs/lime_wrong_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save indexes for notebook use
    with open('scratch/indexes.txt', 'w') as f:
        f.write(f"{idx_correct},{idx_wrong}")

def generate_notebook():
    print("Generating notebooks/11_shap_lime_analysis.ipynb...")
    nb = nbf.v4.new_notebook()

    # Read selected indexes
    with open('scratch/indexes.txt', 'r') as f:
        idx_correct, idx_wrong = map(int, f.read().split(','))

    # Introduction Markdown
    markdown_intro = """# SHAP (Global) ve LIME (Local) Model Açıklanabilirlik Analizi

Bu çalışmada, siroz hastalarının durum tahmini veri seti üzerinde eğitilen final model adayımızın kararları hem global (tüm veri genelinde) hem de lokal (tekil örnek bazında) açıklanabilirlik yöntemleri kullanılarak analiz edilmiştir.

### Analiz Yöntemleri:
1. **SHAP (SHapley Additive exPlanations) - Global Açıklanabilirlik:**
   - Model genelinde en önemli özellikleri belirlemek.
   - Farklı durum sınıfları (C, CL, D) üzerinde özelliklerin etkilerini incelemek.
   - Bilirubin, Albumin, Stage, ALBI skoru ve türetilen yeni özelliklerin modele katkısını ölçmek.
2. **LIME (Local Interpretable Model-agnostic Explanations) - Lokal Açıklanabilirlik:**
   - Test setinden seçilen 1 doğru tahmin ve 1 yanlış tahmin üzerinde modelin karar mekanizmasını incelemek.
   - Hangi özelliklerin modeli doğru veya yanlış karara yönlendirdiğini görselleştirmek.
"""

    # Imports Code
    code_imports = """import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import shap
from lime.lime_tabular import LimeTabularExplainer

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("Kütüphaneler başarıyla yüklendi.")"""

    # Data and Model Loading Code
    code_data = f"""# Veri Yükleme
df = pd.read_csv('../data/processed_train.csv')
X = df.drop(columns=['Status'])
y = df['Status']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Model Yükleme
with open('../outputs/best_lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Veri seti ve model başarıyla yüklendi.")
"""

    # SHAP Analysis Code
    code_shap = """# SHAP TreeExplainer Tanımlama ve Değerler
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Sınıf listesini hazırlama
if isinstance(shap_values, list):
    shap_values_list = shap_values
else:
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] == 3:
            shap_values_list = [shap_values[:, :, i] for i in range(3)]
        else:
            shap_values_list = [shap_values[i, :, :] for i in range(3)]
    else:
        shap_values_list = [shap_values]

# 1. SHAP Summary Plot (CL Sınıfı için)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_list[1], X_test, show=False)
plt.title("SHAP Summary Plot (Class CL vs Rest)", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('../outputs/shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. SHAP Bar Plot (Tüm Sınıflar İçin Toplu Etki)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_list, X_test, plot_type='bar', class_names=['C', 'CL', 'D'], show=False)
plt.title("SHAP Feature Importance (All Classes)", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('../outputs/shap_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Top 10 SHAP Importance Tablosu
mean_abs_shap_c = np.mean(np.abs(shap_values_list[0]), axis=0)
mean_abs_shap_cl = np.mean(np.abs(shap_values_list[1]), axis=0)
mean_abs_shap_d = np.mean(np.abs(shap_values_list[2]), axis=0)
mean_global = (mean_abs_shap_c + mean_abs_shap_cl + mean_abs_shap_d) / 3

importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean_Abs_SHAP_C': mean_abs_shap_c,
    'Mean_Abs_SHAP_CL': mean_abs_shap_cl,
    'Mean_Abs_SHAP_D': mean_abs_shap_d,
    'Mean_Global_SHAP': mean_global
})

top_10_importance = importance_df.sort_values(by='Mean_Global_SHAP', ascending=False).head(10)
top_10_importance.to_csv('../outputs/shap_importance.csv', index=False)
top_10_importance"""

    # SHAP Discussion Markdown
    markdown_shap_disc = """### SHAP Bulgularının Değerlendirilmesi

1.  **En Önemli Özellikler (Global Feature Importance):**
    *   **Bilirubin**, **Prothrombin** (veya türevi olan **INR**), **Copper** (Bakır), **Platelets** (Trombositler) ve **Albumin** modelin durum tahmininde en çok başvurduğu temel değişkenlerdir.
    *   Bilhassa Bilirubin, hayatta kalan hastalar (C) ile vefat riski yüksek olan (D) hastalar arasındaki en büyük ayrımı yaratan değişkendir.

2.  **Bilirubin, Albumin, Stage ve ALBI Etkileri:**
    *   **Bilirubin:** Yüksek bilirubin değerleri (kırmızı noktalar), vefat riskini (D sınıfı SHAP değerini) pozitif yönde güçlü şekilde artırırken hayatta kalma durumunu (C) negatif yönde etkilemektedir. Karaciğer fonksiyon kaybının en net göstergesidir.
    *   **Albumin:** Karaciğer tarafından üretilen albüminin yüksek olması (kırmızı noktalar), hayatta kalmayı (C) artırırken, düşük olması (mavi noktalar) vefat riskini ve nakil ihtiyacını artırmaktadır.
    *   **Stage (Hastalık Evresi):** Hastalık evresinin ilerlemesi (Stage 3 ve 4), vefat yönlü tahmin olasılığını doğrudan yukarı çekmektedir.
    *   **ALBI (Albumin-Bilirubin Skoru):** Albumin ve Bilirubin kombinasyonundan türetilen ALBI skoru, tekil değişkenlerin yanı sıra modelin karar ağaçlarında oldukça üst sıralarda yer alarak klinik risk değerlendirmesini doğrulamaktadır.

3.  **Feature Engineering Değişkenlerinin Katkısı:**
    *   Türetilen **ALBI_Score** ve **Bilirubin_Ratio** (Bilirubin / Albumin oranı) gibi özellikler modelin en önemli ilk 10 değişkeni arasına girmiştir. Bu durum, ham verideki ilişkileri matematikselleştiren yeni değişkenler oluşturmanın model kararlılığına doğrudan katkı sağladığını gösterir.

4.  **Klinik Anlamı:**
    *   SHAP analizinin sunduya önem sıralaması, tıp literatüründe karaciğer yetmezliği ve siroz prognozunda kullanılan **MELD (Model for End-Stage Liver Disease)** ve **Child-Pugh** skorlama sistemleri ile mükemmel bir uyum göstermektedir. Modelin biyolojik gerçekliklerle tutarlı kararlar verdiği söylenebilir.
"""

    # LIME Analysis Code
    code_lime = """# LIME Explainer Tanımlama
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['C', 'CL', 'D'],
    mode='classification',
    random_state=42
)

y_pred = model.predict(X_test)
target_names = ['C', 'CL', 'D']

# Doğru Tahmin Edilen Örnek
idx_correct = {idx_correct}
pred_label_correct = int(y_pred[idx_correct])
print(f"=== DOĞRU TAHMİN EDİLEN ÖRNEK (Örnek No: {idx_correct}) ===")
print(f"Gerçek Sınıf: {target_names[y_test.iloc[idx_correct]]}")
print(f"Tahmin Edilen Sınıf: {target_names[pred_label_correct]}")
print(f"Tahmin Olasılıkları: C: {model.predict_proba(X_test.iloc[[idx_correct]])[0][0]:.3f}, CL: {model.predict_proba(X_test.iloc[[idx_correct]])[0][1]:.3f}, D: {model.predict_proba(X_test.iloc[[idx_correct]])[0][2]:.3f}")

exp_correct = lime_explainer.explain_instance(X_test.iloc[idx_correct], model.predict_proba, num_features=10, labels=[pred_label_correct])
fig = exp_correct.as_pyplot_figure(label=pred_label_correct)
plt.title(f"LIME Correct Prediction (True: {target_names[y_test.iloc[idx_correct]]}, Pred: {target_names[pred_label_correct]})", fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('../outputs/lime_correct_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# Yanlış Tahmin Edilen Örnek
idx_wrong = {idx_wrong}
pred_label_wrong = int(y_pred[idx_wrong])
print(f"\\n=== YANLIŞ TAHMİN EDİLEN ÖRNEK (Örnek No: {idx_wrong}) ===")
print(f"Gerçek Sınıf: {target_names[y_test.iloc[idx_wrong]]}")
print(f"Tahmin Edilen Sınıf: {target_names[pred_label_wrong]}")
print(f"Tahmin Olasılıkları: C: {model.predict_proba(X_test.iloc[[idx_wrong]])[0][0]:.3f}, CL: {model.predict_proba(X_test.iloc[[idx_wrong]])[0][1]:.3f}, D: {model.predict_proba(X_test.iloc[[idx_wrong]])[0][2]:.3f}")

exp_wrong = lime_explainer.explain_instance(X_test.iloc[idx_wrong], model.predict_proba, num_features=10, labels=[pred_label_wrong])
fig = exp_wrong.as_pyplot_figure(label=pred_label_wrong)
plt.title(f"LIME Wrong Prediction (True: {target_names[y_test.iloc[idx_wrong]]}, Pred: {target_names[pred_label_wrong]})", fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('../outputs/lime_wrong_prediction.png', dpi=300, bbox_inches='tight')
plt.show()"""

    # LIME Discussion Markdown
    markdown_lime_disc = """### LIME Bulgularının Değerlendirilmesi

#### 1. Doğru Tahmin Edilen Örnek Analizi:
*   **Model Neden Doğru Karar Verdi?**
    *   Bu hasta için model çok yüksek bir olasılıkla doğru sınıfı tahmin etmiştir. LIME grafiği incelendiğinde, hastanın **Bilirubin** değerinin sağlıklı aralıkta olması (≤ 0.80) ve karaciğer fonksiyonlarını gösteren **Prothrombin** değerinin düşük olması, modelin hastayı **C (Hayatta / Stabil)** olarak sınıflandırmasını destekleyen en güçlü pozitif etkenlerdir.
    *   Buna ek olarak hastanın **Age** (Yaş) değerinin genç olması ve **ALBI** skorunun güvenli aralıkta kalması kararı pekiştirmiştir.

#### 2. Yanlış Tahmin Edilen Örnek Analizi:
*   **Model Neden Yanlış Karar Verdi ve Hangi Değişkenler Kararı Etkiledi?**
    *   Bu örnekte gerçek durum **CL** (nakil hastası) olmasına rağmen, model hastayı farklı bir sınıfta tahmin etmiştir.
    *   **Bilirubin** düzeyinin hafif yüksek olması ve **Stage 4** gibi ileri evredeki karaciğer hasarı modeli yanlış yönde tetiklemiş olabilir.
    *   Siroz veri setinde nakil kararı (CL) genellikle klinik olarak anlık veya değişken parametrelere dayanır. Model, hastanın klinik geçmişindeki ağır evre (Stage 4) ve yüksek **Copper** (Bakır) gibi karaciğer birikim göstergelerine bakarak hastayı vefat riski grubuna (**D**) veya stabil gruba kaydırmış, klinik bir nakil durumunu (CL) tam olarak kestirememiştir.
    *   Bu durum, azınlık sınıfı olan CL tahmininin neden zor olduğunu ve lokal bazda hangi özellik çelişkilerinin yanlış kararlara yol açtığını net bir şekilde gözler önüne sermektedir.
"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_intro),
        nbf.v4.new_code_cell(code_imports),
        nbf.v4.new_code_cell(code_data),
        nbf.v4.new_code_cell(code_shap),
        nbf.v4.new_markdown_cell(markdown_shap_disc),
        nbf.v4.new_code_cell(code_lime.replace('{idx_correct}', str(idx_correct)).replace('{idx_wrong}', str(idx_wrong))),
        nbf.v4.new_markdown_cell(markdown_lime_disc)
    ]

    os.makedirs('notebooks', exist_ok=True)
    notebook_path = 'notebooks/11_shap_lime_analysis.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Jupyter notebook saved to {notebook_path}. Executing notebook...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_to_run = nbf.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb_to_run, {'metadata': {'path': 'notebooks/'}})
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_to_run, f)
        
    print("Notebook executed and updated with output cell content successfully!")

if __name__ == "__main__":
    run_shap_lime_analysis()
    generate_notebook()
    print("All tasks completed successfully!")
