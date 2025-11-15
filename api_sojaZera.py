import os
import io
import joblib
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from skimage.feature import hog, local_binary_pattern
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch
from transformers import ViTModel, ViTImageProcessor

# === ngrok ===
from pyngrok import ngrok

# =========================================
# 🚀 CONFIGURAÇÃO
# =========================================
app = Flask(__name__, static_folder=".", static_url_path="")

BASE_PATH = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo"

# Caminhos dos modelos
SVM_HOGLBP_PATH = os.path.join(BASE_PATH, "modelo_hoglbp_aprimorado_svm_rbf.pkl")
SCALER_PATH = os.path.join(BASE_PATH, "scaler_hoglbp_aprimorado.pkl")
PCA_PATH = os.path.join(BASE_PATH, "pca_hoglbp_aprimorado.pkl")
CLASSES_PATH = os.path.join(BASE_PATH, "classes.pkl")

RF_RESNET_PATH = os.path.join(BASE_PATH, "randomforest_resnet_model.pkl")
SCALER_RESNET_PATH = os.path.join(BASE_PATH, "resnet_scaler.pkl")
PCA_RESNET_PATH = os.path.join(BASE_PATH, "resnet_pca.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(BASE_PATH, "resnet_feature_extractor.h5")
CLASSES_RESNET_PATH = os.path.join(BASE_PATH, "classes.npy")

VIT_BASE_PATH = os.path.join(BASE_PATH, "modelo_vit/vit_feature_extractor")
VIT_RF_PATH = os.path.join(BASE_PATH, "modelo_vit/vit_randomforest.pkl")
VIT_SCALER_PATH = os.path.join(BASE_PATH, "modelo_vit/vit_scaler.pkl")
VIT_PCA_PATH = os.path.join(BASE_PATH, "modelo_vit/vit_pca.pkl")
VIT_CLASSES_PATH = os.path.join(BASE_PATH, "modelo_vit/classes.npy")

# =========================================
# 🧠 CARREGAMENTO DOS MODELOS
# =========================================
print("🔄 Carregando modelos e pré-processadores...")

# SVM + HOG/LBP
svm_hoglbp = joblib.load(SVM_HOGLBP_PATH)
scaler_hoglbp = joblib.load(SCALER_PATH)
pca_hoglbp = joblib.load(PCA_PATH)
if os.path.exists(CLASSES_PATH):
    CLASSES = joblib.load(CLASSES_PATH)
else:
    CLASSES = ['R1_R2','R5_R6','R7_R8','V1_V2','V3_V4','VE_VC']

# Random Forest + ResNet
rf_resnet = joblib.load(RF_RESNET_PATH)
scaler_resnet = joblib.load(SCALER_RESNET_PATH)
pca_resnet = joblib.load(PCA_RESNET_PATH)
feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
if os.path.exists(CLASSES_RESNET_PATH):
    CLASSES_RF = np.load(CLASSES_RESNET_PATH)
else:
    CLASSES_RF = CLASSES

print("🔄 Carregando modelo Vision Transformer...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar processador e modelo ViT
vit_processor = ViTImageProcessor.from_pretrained(VIT_BASE_PATH)
vit_model = ViTModel.from_pretrained(VIT_BASE_PATH).to(device)
vit_model.eval()

# Carregar Random Forest, PCA e Scaler
rf_vit = joblib.load(VIT_RF_PATH)
scaler_vit = joblib.load(VIT_SCALER_PATH)
pca_vit = joblib.load(VIT_PCA_PATH)

# Carregar classes
if os.path.exists(VIT_CLASSES_PATH):
    CLASSES_VIT = np.load(VIT_CLASSES_PATH, allow_pickle=True)
else:
    CLASSES_VIT = CLASSES_RF

print("✅ Vision Transformer carregado com sucesso!")
print("✅ Todos os modelos carregados!")

# =========================================
# 🧮 FUNÇÕES AUXILIARES
# =========================================
def extract_vit_features(image_cv):
    """Extrai embeddings CLS do Vision Transformer."""
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    inputs = vit_processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
    return features

def read_image_file(stream):
    return Image.open(stream).convert('RGB')

def extract_features_enhanced(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    features = []

    hog_features, _ = hog(gray, orientations=18, pixels_per_cell=(16,16),
                          cells_per_block=(2,2), block_norm='L2-Hys',
                          visualize=True, transform_sqrt=True, feature_vector=True)
    features.extend(hog_features)

    for radius in [1,2,3]:
        n_points = 8*radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), range=(0,n_points+2))
        lbp_hist = lbp_hist.astype('float') / (lbp_hist.sum()+1e-6)
        features.extend(lbp_hist)

    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    for i in range(3):
        ch = hsv[:,:,i]
        features.extend([np.mean(ch), np.std(ch), np.median(ch)])
    for i in range(3):
        hist = cv2.calcHist([hsv],[i],None,[16],[0,256])
        hist = cv2.normalize(hist,hist).flatten()
        features.extend(hist)

    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([np.mean(grad_mag), np.std(grad_mag)])

    return np.array(features).reshape(1,-1)

def extract_resnet_features(image_cv):
    img_resized = cv2.resize(image_cv,(224,224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_preprocessed = preprocess_input(img_array)
    feats = feature_extractor.predict(img_preprocessed, verbose=0)
    return feats.flatten().reshape(1,-1)

# =========================================
# 🌐 ROTAS
# =========================================
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/predict", methods=["POST"])
def predict_both():
    if 'file' not in request.files:
        return jsonify({"error": 'Envie a imagem no campo "file".'}),400

    file = request.files['file']
    try:
        pil_img = read_image_file(file.stream)
        image_cv = np.array(pil_img)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": f"Erro ao ler imagem: {e}"}),400

    # SVM + HOG/LBP
    feats_svm = extract_features_enhanced(cv2.resize(image_cv,(128,128)))
    feats_svm = scaler_hoglbp.transform(feats_svm)
    feats_svm = pca_hoglbp.transform(feats_svm)
    probs_svm = svm_hoglbp.predict_proba(feats_svm)[0]
    idx_svm = int(np.argmax(probs_svm))
    result_svm = {
        "classe_prevista": CLASSES[idx_svm],
        "acuracia": round(float(probs_svm[idx_svm])*100,2),
        "percentuais": {CLASSES[i]: round(float(probs_svm[i])*100,2) for i in range(len(CLASSES))}
    }

    # Random Forest + ResNet
    feats_rf = extract_resnet_features(image_cv)
    feats_rf = scaler_resnet.transform(feats_rf)
    feats_rf = pca_resnet.transform(feats_rf)
    probs_rf = rf_resnet.predict_proba(feats_rf)[0]
    idx_rf = int(np.argmax(probs_rf))
    result_rf = {
        "classe_prevista": CLASSES_RF[idx_rf],
        "acuracia": round(float(probs_rf[idx_rf])*100,2),
        "percentuais": {CLASSES_RF[i]: round(float(probs_rf[i])*100,2) for i in range(len(CLASSES_RF))}
    }
    # Vision Transformer + Random Forest
    feats_vit = extract_vit_features(image_cv)
    feats_vit = scaler_vit.transform(feats_vit)
    feats_vit = pca_vit.transform(feats_vit)
    probs_vit = rf_vit.predict_proba(feats_vit)[0]
    idx_vit = int(np.argmax(probs_vit))
    result_vit = {
        "classe_prevista": CLASSES_VIT[idx_vit],
        "acuracia": round(float(probs_vit[idx_vit]) * 100, 2),
        "percentuais": {CLASSES_VIT[i]: round(float(probs_vit[i]) * 100, 2) for i in range(len(CLASSES_VIT))}
    }

    print(">>> RESULTADO ViT:", result_vit)
    print(">>> RESULTADO SVM:", result_svm)
    print(">>> RESULTADO RF ResNet:", result_rf)

    return jsonify({
        "svm_hoglbp": result_svm,
        "rf_resnet": result_rf,
        "vit_randomforest": result_vit
    })

# =========================================
# ▶️ EXECUÇÃO LOCAL COM NGROK
# =========================================
if __name__ == "__main__":
    # Abrir túnel ngrok para porta 5000
    public_url = ngrok.connect(5000)
    print("🔗 URL pública ngrok:", public_url)

    app.run(port=5000)
