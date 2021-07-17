import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = pd.read_csv('dataset.csv').values
X = data[:,:2].reshape(-1,2)
y = data[:,2].reshape(-1,1)

plt.scatter(X[:10, 0], X[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(X[10:, 0], X[10:, 1], c='blue', edgecolors='none', s=30, label='từ chối')

plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')


N = data.shape[0]
X = np.hstack((np.ones((N,1)),X))

w = np.array([0., 0.1, 0.1]).reshape(-1,1)
w = np.hstack((w,w,w,w,w))

numOfIters = 1000

lr_lst = [0.001, 0.003, 0.01, 0.03, 0.1]

for idx, lr in enumerate(lr_lst):
    for i in range(numOfIters):
        m = w[:,idx].reshape(-1,1)
        y_pred = sigmoid(np.dot(X,m))
        m -= lr * np.dot(X.T, y_pred - y)
        w[:,idx] = m.reshape(1,-1)


print(w)
for i in range(5):
    plt.plot([4,10],[-(w[0,i] + w[1,i]*4)/w[2,i],-(w[0,i] + w[1,i]*10)/w[2,i]],label = 'lr = %s'%lr_lst[i])
plt.legend()
plt.show()