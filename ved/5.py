import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [0], [1], [0]])

np.random.seed(1)
wh = np.random.uniform(-1, 1, (2, 2))
bh = np.random.uniform(-1, 1, (1, 2))
wo = np.random.uniform(-1, 1, (2, 1))
bo = np.random.uniform(-1, 1, (1, 1))

epochs = 10000
lr = 0.1

for i in range(epochs):
    #forward
    hi = np.dot(x,wh)+bh
    ho = sigmoid(hi)
    
    fi = np.dot(ho,wo)+bo
    out = sigmoid(fi)
    
    #backward
    e = y-out
    do = e*sigmoid_derivative(out)
    
    eh = do.dot(wo.T)
    dh = eh*sigmoid_derivative(ho)
   
   
    
    #update
    wo += ho.T.dot(do) * lr
    bo += np.sum(do, axis=0, keepdims=True) * lr
    wh += x.T.dot(dh) * lr
    bh += np.sum(dh, axis=0, keepdims=True) * lr
print(out)