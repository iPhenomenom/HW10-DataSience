import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# Графік точності та втрат під час навчання
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Завантаження та підготовка даних
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Збільшення розміру зображень з 28x28 до 48x48 та додавання каналу RGB
train_images = tf.image.grayscale_to_rgb(tf.image.resize(train_images, (48, 48)))
test_images = tf.image.grayscale_to_rgb(tf.image.resize(test_images, (48, 48)))

# Масштабування пікселів до діапазону [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Згорткова основа VGG16
vgg_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(48, 48, 3))  # Змініть розмір входу на (48, 48, 3)

# Створення моделі
model = models.Sequential()
model.add(vgg_base)  # Додавання VGG16 як згорткової основи
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Замороження згорткової основи (опціонально)
# vgg_base.trainable = False

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Оцінка та візуалізація результатів
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# Графік точності та втрат під час навчання

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
