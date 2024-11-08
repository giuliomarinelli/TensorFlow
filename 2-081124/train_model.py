import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
index = 4

# # Set number of characters per row when printing
# np.set_printoptions(linewidth=320)

# # Print the label and image
# print(f'LABEL: {training_labels[index]}')
# print(f'\nIMAGE PIXEL ARRAY:\n\n{training_images[index]}\n\n')

# # Visualize the image using the default colormap (viridis)
# plt.imshow(training_images[index])
# plt.colorbar()
# plt.show()

training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Declare sample inputs and convert to a tensor
# inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
# inputs = tf.convert_to_tensor(inputs)
# print(f'input to softmax function: {inputs.numpy()}')

# # Feed the inputs to a softmax activation function
# outputs = tf.keras.activations.softmax(inputs)
# print(f'output of softmax function: {outputs.numpy()}')

# # Get the sum of all values after the softmax
# sum = tf.reduce_sum(outputs)
# print(f'sum of outputs: {sum}')

# # Get the index with highest value
# prediction = np.argmax(outputs)
# print(f'class with highest probability: {prediction}')

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
model.save('fashion_mnist_model.keras')