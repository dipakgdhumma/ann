import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 100)

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# Plot Sigmoid
plt.figure(figsize=(6, 4))
plt.plot(x, sigmoid(x), color='blue')
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Plot Tanh
plt.figure(figsize=(6, 4))
plt.plot(x, tanh(x), color='green')
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Plot ReLU
plt.figure(figsize=(6, 4))
plt.plot(x, relu(x), color='red')
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Plot Leaky ReLU
plt.figure(figsize=(6, 4))
plt.plot(x, leaky_relu(x), color='purple')
plt.title("Leaky ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
