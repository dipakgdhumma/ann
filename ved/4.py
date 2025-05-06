import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

x = np.array([
    [1, 1],  # Class 0
    [2, 1],  # Class 0
    [3, 1],  # Class 0
    [4, 1],  # Class 0
    [1, 4],  # Class 1
    [2, 4],  # Class 1
    [3, 4],  # Class 1
    [4, 4],  # Class 1
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = Perceptron(max_iter=1000, eta0=0.01)
model.fit(x, y)

for i in range(len(y)):
    if y[i]==0:
        plt.scatter(x[i][0], x[i][1], color='red', label='class 0' if i==0 else "")
    else:
        plt.scatter(x[i][0], x[i][1], color='blue', label='class 1' if i==0 else "")
        
w = model.coef_[0]
b = model.intercept_

x_val = np.linspace(0, 6, 100)
y = -((w[0]*x_val)+b)/w[1]

plt.plot(x_val, y, 'k-.', label='Decision Boundary')
plt.xlabel('x1')
plt.xlabel('x2')
plt.legend()
plt.grid()


sp = np.array([
    [3, 2],
    [2 , 3]
])

for i in sp:
    pred = model.predict([i])
    print(f"{i} : {pred}")
    if pred == 0:
        plt.scatter(i[0], i[1], marker='*', color='red')
    else:
        plt.scatter(i[0], i[1], marker='*', color='blue')
        
plt.show()