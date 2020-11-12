import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg

#Parameter Initialization

mu = 0.01
a = 0.99
num_ite = 100
num_data = 5000
mult = 200
var = 0.019853
sigu2 = 0.93627
sigv2 = (1-(a**2)) * sigu2
a2 = a**2
t = np.asarray(range(1, 5001))
W_all = None
error_all = None
p5mu = 0.5*mu

#LMS main process

decay = 0
Npred = 5000
Xi0 = 0


for i1 in range(100):

    l=np.random.normal(0, math.sqrt(var), size=5000)

    l2 = [i2 for i2 in range(5000)]
    l2[0] = l[0]
    l2[1] = a * l2[0] + l[1]

    for i3 in range(2, 5000):

        l2[i3] = a*l2[i3-1] + l[i3]

    l2 = np.asarray(l2).reshape(5000, 1)

#Predict

    N = l2.shape[0]
    W_1 = []
    W = 0
    n = 0
    W_1.append(W)
    W_1 = np.asarray(W_1)
    data_predict = np.zeros((5000, 1))
    error = np.zeros((5000, 1))
    data_error = np.zeros((5000, 1))
    data_predict[n, :] = Xi0 * W
    error[n, :] = l2[n, :] - data_predict[n, :]
    data_error[n] = linalg.norm(error[n, :])
    W = W + mu * error[n, :].T * Xi0

    for i4 in range(1, N):
        W_1 = np.append(W_1, W)
        data_predict[i4, :] = l2[i4-1, :] * W
        error[i4, :] = l2[i4, :] - data_predict[i4, :]
        data_error[i4] = linalg.norm(error[i4, :])
        W = W + mu * error[i4, :].T * l2[i4-1, :]

    if i1 == 0:
        W_all = np.copy(W_1)
        error_all = np.copy(error)

    else:
        W_all = np.concatenate((W_all, np.copy(W_1)), axis=0)
        error_all = np.concatenate((error_all, np.copy(error)), axis=0)

W_all = W_all.reshape(100, 5000)
error_all = error_all.reshape((100, 5000))
E = np.square(error_all)
E_mean = np.mean(E, axis=0)

J = sigu2*(1-a**2)*(1+(mu/2)*sigu2) + sigu2*(a**2+(mu/2)*(a**2)*sigu2-0.5*mu*sigu2)*(1-mu*sigu2)**(2*t)

plt.plot(J, color='red')
plt.semilogy(E_mean, 'k--')
plt.xlabel('Number of iterations')
plt.ylabel('MSE')
plt.show()






