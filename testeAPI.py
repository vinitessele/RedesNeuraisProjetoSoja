# ============================================================
# 🚀 API Flask + PyTorch + Ngrok — Classificação Estágios da Soja
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import os
import timm
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from pyngrok import ngrok, conf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PATHS
# ============================================================
UPLOAD_FOLDER = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\imagens\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESNET_PATH = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo\modeloZera\resnet50_modeloZera.pth"
VIT_PATH = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo\modeloZera\vit_modeloZera.pth"
EFFICIENTNET_PATH = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo\modeloZera\efficientnetv2_modeloZera.pth"
SWIN_PATH = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo\modeloZera\swin_modeloZera.pth"
VIT_PATH2 = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo\modeloZeraIndividual\vit_modeloZera.pth"

CLASSES = ['R1_R2', 'R5_R6', 'R7_R8', 'V1_V2', 'V3_V4', 'VE_VC']
CLASSES2 = ['R1','R2','R3','R4', 'R5','R6', 'R7','R8','V1','V2', 'V3','V4','V5', 'VE','VC']

# ============================================================
# CARREGAR MODELOS
# ============================================================

# ---- ResNet50 ----
resnet = models.resnet50(weights=None)
num_ftrs_resnet = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs_resnet, len(CLASSES))
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
resnet.eval().to(DEVICE)

# ---- ViT Base ----
vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASSES))
vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
vit.eval().to(DEVICE)

# ---- ViT Base ----
vit2 = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASSES2))
vit2.load_state_dict(torch.load(VIT_PATH2, map_location=DEVICE))
vit2.eval().to(DEVICE)

# ---- EfficientNetV2 ----
efficientnet = efficientnet_v2_s(weights=None)
num_ftrs = efficientnet.classifier[1].in_features
efficientnet.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))
efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=DEVICE))
efficientnet.eval().to(DEVICE)

# ---- Swin Transformer ----
swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=len(CLASSES)
)
swin.load_state_dict(torch.load(SWIN_PATH, map_location=DEVICE))
swin.eval().to(DEVICE)


# ============================================================
# TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# FUNÇÕES
# ============================================================
def predict_with_model(model, image_tensor):
    with torch.no_grad():
        out = model(image_tensor)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = probs.argmax()
        return CLASSES[idx], {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

def predict_with_model2(model, image_tensor):
    with torch.no_grad():
        out = model(image_tensor)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = probs.argmax()
        return CLASSES2[idx], {CLASSES2[i]: float(probs[i]) for i in range(len(CLASSES2))}
    
def predict_image_all(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        return {"error": "Arquivo enviado não é uma imagem válida."}

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    r_pred, r_probs = predict_with_model(resnet, img_tensor)
    v_pred, v_probs = predict_with_model(vit, img_tensor)
    e_pred, e_probs = predict_with_model(efficientnet, img_tensor)
    s_pred, s_probs = predict_with_model(swin, img_tensor)
    v_pred2, v_probs2 = predict_with_model2(vit2, img_tensor)

    return {
        "2vit_base_patch16_224": {"predicted_class": v_pred2, "probabilities": v_probs2},
        "resnet50": {"predicted_class": r_pred, "probabilities": r_probs},
        "vit_base_patch16_224": {"predicted_class": v_pred, "probabilities": v_probs},
        "efficientnet_v2_l": {"predicted_class": e_pred, "probabilities": e_probs},
        "swin_tiny_patch4_window7_224": {"predicted_class": s_pred, "probabilities": s_probs},
    }


# ============================================================
# API FLASK
# ============================================================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Envie um arquivo usando o campo 'file'"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    # Realiza predição com todos os modelos
    result = predict_image_all(image_bytes)

    # 📌 Usa EfficientNet como predição principal
    predicted_class = result["vit_base_patch16_224"]["predicted_class"]

    # Novo nome do arquivo
    name, ext = os.path.splitext(file.filename)
    new_filename = f"{name}__{predicted_class}{ext}"

    # Salvar imagem com novo nome
    save_path = os.path.join(UPLOAD_FOLDER, new_filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    print(f"📁 Imagem salva como: {new_filename}")
    print(f"Result:",result)
    
    return jsonify({
        "original_filename": file.filename,
        "saved_as": new_filename,
        "predictions": result
    })


@app.route("/")
def index():
    return send_from_directory('.', 'index.html')


# ============================================================
# NGROK COM TOKEN 🚀
# ============================================================
NGROK_TOKEN = "2rv3sOP99V8s5Onjx3V2PTD0LoP_3HutrBirTFVfWFjY9oLei"

conf.get_default().auth_token = NGROK_TOKEN

print("🔌 Iniciando ngrok autenticado...")
public_url = ngrok.connect(5000)
print(f"🌍 URL pública da API: {public_url}")


# ============================================================
# RODAR SERVIDOR
# ============================================================
if __name__ == "__main__":
    print("🚀 Servidor Flask rodando em http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
