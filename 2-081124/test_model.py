# import_model.py
import tensorflow as tf
import numpy as np

# Carico il dataset
fmnist = tf.keras.datasets.fashion_mnist
# Carica il modello già addestrato
model = tf.keras.models.load_model('fashion_mnist_model.keras')

# Carica il dataset di test
(_, _), (test_images, test_labels) = fmnist.load_data()
test_images = test_images / 255.0  # Normalizza le immagini

# Valuta il modello sui dati di test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Prevedi una singola immagine (esempio: indice 0)
index = 0
prediction = model.predict(np.expand_dims(test_images[index], axis=0))
predicted_label = np.argmax(prediction)
print(f'Predicted label for test image at index {index}: {predicted_label}')
print(f'Actual label: {test_labels[index]}')
