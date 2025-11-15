import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io

# Inicializando o Flask
app = Flask(__name__)

# Carregar o modelo treinado
model = tf.keras.models.load_model('saved_models/simple_cnn_model.h5')

# Obtém a forma de entrada esperada pelo modelo
input_shape = model.input_shape[1:4]  # Ignora o primeiro elemento (batch size)
if len(input_shape) != 3:
    raise ValueError("O modelo deve ter entrada 3D (altura, largura, canais).")

# Função para pré-processar a imagem
def preprocess_image(image):
    # Redimensiona para o tamanho esperado pelo modelo
    image = image.resize((input_shape[0], input_shape[1]))
    # Converte para array numpy e normaliza
    image_array = np.array(image) / 255.0
    # Verifica se a imagem tem os canais esperados
    if image_array.shape[-1] != input_shape[2]:
        raise ValueError(f"A imagem deve ter {input_shape[2]} canais (RGB esperado).")
    # Adiciona a dimensão de batch
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Rota para receber a imagem e fazer a previsão
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Carrega a imagem usando PIL
        image = Image.open(io.BytesIO(file.read()))
        # Pré-processa a imagem
        image_array = preprocess_image(image)
        # Faz a previsão
        predictions = model.predict(image_array)
        # Obtém a classe com maior probabilidade
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
