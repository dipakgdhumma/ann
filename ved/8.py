import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Input (X) and Output (y) — XOR function
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Seed for reproducibility
np.random.seed(1)

# Initialize weights and biases for 2-2-1 network
wh = np.random.uniform(-1, 1, (2, 2))  # Input to Hidden weights
bh = np.random.uniform(-1, 1, (1, 2))  # Hidden bias

wo = np.random.uniform(-1, 1, (2, 1))  # Hidden to Output weights
bo = np.random.uniform(-1, 1, (1, 1))  # Output bias

# Hyperparameters
lr = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Feedforward
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    final_output = sigmoid(final_input)

    # Backpropagation
    error = y - final_output
    d_output = error * sigmoid_deriv(final_output)

    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_deriv(hidden_output)

    # Update weights and biases
    wo += hidden_output.T.dot(d_output) * lr
    bo += np.sum(d_output, axis=0, keepdims=True) * lr

    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Final Output
print("Predictions after training:")
print(np.round(final_output, 3))

# Optional: Show each input with its prediction
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {round(final_output[i][0], 3)}")
