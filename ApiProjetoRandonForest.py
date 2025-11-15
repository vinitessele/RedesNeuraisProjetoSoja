import io
import numpy as np
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# Carregar o modelo Random Forest
MODEL_PATH = "saved_models/RandomForest.pkl"
rf_model = joblib.load(MODEL_PATH)

# Carregar o modelo ResNet50 para extração de features
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Criar a API com Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """Recebe uma imagem e retorna a previsão do modelo."""
    if 'file' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    # Pré-processar a imagem
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extrair features usando ResNet50
    features = feature_extractor.predict(img_array)
    
    # Fazer a previsão com o modelo Random Forest
    predictions = rf_model.predict_proba(features)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    class_labels = {
        0: "Estádio fenológico menor que R7. Não realizar dessecação.",
        1: (
            "Estádio fenológico maior que R7, lavoura de soja com 75% das folhas "
            "e vagens amarelas. Apto a fazer a dessecação."
        ),
    }
    #class_labels = {0: "estádio fenológico menor que R7, Não realizar dessecação", 1: "estádio fenológico maior R7, lavoura de soja estiver com 76% das folhas e vagens amarelas, Apto a fazer a dessecação"}
    predicted_class_label = class_labels[predicted_class]
    # Retorna a classe prevista

    return jsonify({'predicted_class': predicted_class_label, 'confidence': str(confidence)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)