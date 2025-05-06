import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers

(xtr, ytr), (xte, yte) = datasets.mnist.load_data()

xtr = xtr/255.0
xte = xte/255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
]) 

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xtr, ytr, epochs=5, validation_data=(xte, yte))

test_loss, test_accuracy = model.evaluate(xte, yte)
print(test_accuracy)
pred = model.predict(xte)

plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(xte[i], cmap='gray')
    plt.grid('off')
    plt.title(f"Actual: {yte[i]}\nPredicted: {tf.argmax(pred[i])}")

plt.tight_layout()
plt.show()