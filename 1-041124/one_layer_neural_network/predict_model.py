import tensorflow as tf
import numpy as np

# Caricamento del modello già addestrato
model = tf.keras.models.load_model('model.keras')

# Predizione
predizione = model.predict(np.array([10.0]), verbose=0).item()
print(f"\nNumero predetto dal modello: {predizione:.5f}\n")
