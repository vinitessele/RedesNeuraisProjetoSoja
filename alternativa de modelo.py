import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Parâmetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# Diretório com as imagens
TRAIN_DIR = 'D:\\dataset\\train'
VALID_DIR = 'D:\\dataset\\validation'

# Pré-processamento das imagens
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
valid_generator = valid_datagen.flow_from_directory(VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')


# Construção da CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
history = model.fit(train_generator, validation_data=valid_generator, epochs=EPOCHS)

# Avaliação final no conjunto de validação
val_loss, val_acc = model.evaluate(valid_generator)
print(f"Validation Accuracy: {val_acc:.2f}")

# Salvando o modelo
model.save("modelo_soja.h5")


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Previsões no conjunto de validação
y_pred = model.predict(valid_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = valid_generator.classes

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred_classes)

# Valores absolutos
plt.figure(figsize=(10, 8))
disp_abs = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_generator.class_indices.keys())
disp_abs.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title('Matriz de Confusão - Valores Absolutos', fontsize=14)
plt.show()

# Valores normalizados
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=valid_generator.class_indices.keys())
disp_norm.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title('Matriz de Confusão - Porcentagem (%)', fontsize=14)
plt.show()