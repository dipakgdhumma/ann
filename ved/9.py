import numpy as np

# Define 4 binary patterns (converted to bipolar {-1, 1})
patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

# Number of neurons
n = patterns.shape[1]

# Initialize weight matrix
W = np.zeros((n, n))

# Hebbian Learning Rule
for p in patterns:
    W += np.outer(p, p)

# No self-connections
np.fill_diagonal(W, 0)

print("Weight matrix W:")
print(W)

# Function to update the network
def recall(pattern, steps=5):
    pattern = pattern.copy()
    for _ in range(steps):
        for i in range(n):
            raw = np.dot(W[i], pattern)
            pattern[i] = 1 if raw >= 0 else -1
    return pattern

# Test recall
print("\nTesting recall with original patterns:")
for i, p in enumerate(patterns):
    recalled = recall(p)
    print(f"Pattern {i+1} recalled as: {recalled}")
