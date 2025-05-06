from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

digits = load_digits()
x = digits.data
y_digits = digits.target

y_odd_even = []
for i in range(len(y_digits)):
    k = 1 if (y_digits[i]%2==0) else 0
    y_odd_even.append(k)
    
xtrain, xtest, ytrain, ytest, dtrain, dtest = train_test_split(x, y_odd_even, y_digits, test_size=0.2)

model = Perceptron(max_iter=1000, eta0=0.1)
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)

ac = accuracy_score(y_pred, ytest)

print(ac)

for i in range(15):
    print(f"{dtest[i]} || P{y_pred[i]} || A{ytest[i]}")