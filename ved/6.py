import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report, confusion_matrix

(xtr, ytr), (xte, yte) = datasets.cifar10.load_data()

ytr = ytr.reshape(-1, )
yte = yte.reshape(-1, )

xtr = xtr/255.0
xte = xte/255.0

classes = ['airplane', 'automobile', 'truck', 'ship','dog','deer','frog', 'cat', 'horse', 'bird']

model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer="SGD", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xtr, ytr, epochs=5)

ypred = model.predict(xte)
ypred_classes = []
for e in ypred:
    max_ind=np.argmax(e)
    ypred_classes.append(max_ind)
    
plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(xte[i])
    actual = classes[yte[i]]    
    predicted = classes[ypred_classes[i]]
    plt.title(f"Actual: {actual}\nPredicted: {predicted}")
    
plt.tight_layout()
plt.show()    

