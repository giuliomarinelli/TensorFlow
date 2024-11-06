import tensorflow as tf
import numpy as np

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

print("Compilazione del modello in corso")

model.compile(optimizer='adam', loss='mean_squared_error')

print("Addestramento del modello in corso")
# Addestramento del modello
model.fit(xs, ys, epochs=1000, verbose=0)

# Salvataggio del modello addestrato
model.save('model.keras')
print("Modello addestrato e salvato come 'model.keras'")
