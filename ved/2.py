# Mcculloh Pitts ANDNOT

import numpy as np

def mcp(input, weights, threshold):
    sum = np.dot(input, weights)
    return 1 if sum >= threshold else 0

inputs=[
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

weights = [1, -1]
threshold = 1
print("x1 | x2 || Y")
print("------------")
for input in inputs:
    output = mcp(input, weights, threshold)
    print(f"{input[0]} | {input[1]} || {output}")