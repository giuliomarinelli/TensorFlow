import tensorflow as tf
import numpy as np

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Addestramento del modello
model.fit(xs, ys, epochs=1000, verbose=0)

# Salvataggio del modello addestrato
model.save('il_tuo_modello.h5')
print("Modello addestrato e salvato come 'il_tuo_modello.h5'")
