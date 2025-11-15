from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import io
import timm
import torchvision.transforms as transforms
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Caminhos dos modelos salvos
RESNET_PATH = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo\modeloZera\resnet50_modeloZera.pth"
VIT_PATH = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\modelo\modeloZera\vit_modeloZera.pth"

# Classes
CLASSES = ['R1_R2','R5_R6','R7_R8','V1_V2','V3_V4','VE_VC']

# ===========================
# CARREGAR MODELOS
# ===========================
# ResNet50
import torchvision.models as models
resnet = models.resnet50(weights=None)
num_ftrs_resnet = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs_resnet, len(CLASSES))
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
resnet.eval()
resnet.to(DEVICE)

# ViT Base
vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASSES))
vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
vit.eval()
vit.to(DEVICE)

# ===========================
# TRANSFORM PARA IMAGEM
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===========================
# FUNÇÃO DE PREDIÇÃO COM PROBABILIDADES
# ===========================
def predict_with_model(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = probabilities.argmax()
        predicted_class = CLASSES[predicted_idx]
        class_probabilities = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    return predicted_class, class_probabilities

def predict_image_all(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    resnet_pred, resnet_probs = predict_with_model(resnet, image_tensor)
    vit_pred, vit_probs = predict_with_model(vit, image_tensor)
    
    return {
        "resnet50": {"predicted_class": resnet_pred, "probabilities": resnet_probs},
        "vit_base_patch16_224": {"predicted_class": vit_pred, "probabilities": vit_probs}
    }

# ===========================
# CRIAR API
# ===========================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    file = request.files["file"]
    try:
        image_bytes = file.read()
        predictions = predict_image_all(image_bytes)
        return jsonify({
            "filename": file.filename,
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# RODAR API
# ===========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
