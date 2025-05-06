import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# Set seed for reproducibility
np.random.seed(1)

# Weights and biases (your format)
wh = np.random.uniform(-1, 1, (2, 2))   # weights for input -> hidden
bh = np.random.uniform(-1, 1, (1, 2))   # bias for hidden layer
wo = np.random.uniform(-1, 1, (2, 1))   # weights for hidden -> output
bo = np.random.uniform(-1, 1, (1, 1))   # bias for output layer

# Training parameters
lr = 0.1
epochs = 1000000

# Training loop
for _ in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    final_output = sigmoid(final_input)

    # Backward pass
    error = y - final_output
    d_output = error * sigmoid_deriv(final_output)

    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_deriv(hidden_output)

    # Update weights and biases
    wo += hidden_output.T.dot(d_output) * lr
    bo += np.sum(d_output, axis=0, keepdims=True) * lr

    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Final output
print("Final predictions after training:")
print(np.round(final_output, 3))
