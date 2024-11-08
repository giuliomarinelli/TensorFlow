import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist

# Definisco i nomi delle classi
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Carica il modello già addestrato
model = tf.keras.models.load_model('fashion_mnist_model.keras')

# Carica il dataset di test
(_, _), (test_images, test_labels) = fmnist.load_data()
test_images = test_images / 255.0  # Normalizza le immagini

# Inizializza i contatori per le previsioni corrette e sbagliate per ciascuna classe
correct_counts = np.zeros(10, dtype=int)
incorrect_counts = np.zeros(10, dtype=int)

# Numero di test casuali da effettuare
num_tests = 1000
indices = np.random.choice(len(test_images), size=num_tests, replace=False)

# Loop per effettuare le previsioni
for idx in indices:
    image = np.expand_dims(test_images[idx], axis=0)  # Espandi la dimensione per il batch
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    actual_label = test_labels[idx]
    
    if predicted_label == actual_label:
        correct_counts[actual_label] += 1
    else:
        incorrect_counts[actual_label] += 1

# Crea il grafico a barre per visualizzare i risultati
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(class_names))

# Barre per i conteggi corretti e sbagliati
bars1 = ax.bar(index, correct_counts, bar_width, label='Corrette')
bars2 = ax.bar(index + bar_width, incorrect_counts, bar_width, label='Sbagliate')

# Configurazione del grafico
ax.set_xlabel('Categorie')
ax.set_ylabel('Conteggio')
ax.set_title('Risultati del Test per Categoria')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend()

# Mostra il grafico
plt.tight_layout()
plt.show()
