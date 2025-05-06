import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers

(xtr, ytr), (xte, yte) = datasets.cifar10.load_data()

xtr= xtr/255.0
xte= xte/255.0

ytr = ytr.flatten()
yte = yte.flatten()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer="SGD", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(xtr, ytr, epochs=5, validation_data=(xte, yte))

tl, ta = model.evaluate(xte, yte)

p = model.predict(xte)

plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(xte[i])
    plt.grid('off')
    plt.title(f"Pred: {classes[tf.argmax(p[i])]} \nActual: {classes[yte[i]]}")
plt.show()

