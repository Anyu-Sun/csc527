from random import random
import math
import numpy as np
import matplotlib.pyplot as plt


def moon(num_points, distance, radius, width):
    points = num_points

    x1 = [0 for _ in range(points)]
    y1 = [0 for _ in range(points)]
    x2 = [0 for _ in range(points)]
    y2 = [0 for _ in range(points)]

    for i in range(points):
        d = distance
        r = radius
        w = width
        a = random() * math.pi
        x1[i] = math.sqrt(random()) * math.cos(a) * (w / 2) + (
                    (-(r + w / 2) if (random() < 0.5) else (r + w / 2)) * math.cos(a))
        y1[i] = math.sqrt(random()) * math.sin(a) * (w) + (r * math.sin(a))

        a = random() * math.pi + math.pi
        x2[i] = (r + w / 2) + math.sqrt(random()) * math.cos(a) * (w / 2) + (
            (-(r + w / 2)) if (random() < 0.5) else (r + w / 2)) * math.cos(a)
        y2[i] = -(math.sqrt(random()) * math.sin(a) * (-w) + (-r * math.sin(a))) - d
    return ([x1, x2, y1, y2])

# Decision boundary formula
def draw_line(w, data):

    x = np.linspace(np.amin(data),np.amax(data),100)
    y = -(w[0] + x*w[1])/w[2]
    plt.plot(x, y, '--k',label="DB")

# Get data points
def getDataSet(num_points, distance, radius, width):
    x1, x2, y1, y2 = moon(num_points, distance, radius, width)
    data = []
    data.extend([x1[i], y1[i], 1] for i in range(num_points))
    data.extend([x2[i], y2[i], -1] for i in range(num_points))
    return data

# Signum function, return 1 if activation >= 0 else return -1
def sgnFunc(x, w):
    activation = w[0]
    for i in range(2):
        xx = x[i] * w[i + 1]
        activation = activation + xx
    if activation >= 0:
        return 1
    else:
        return -1

# Create dataset for training
num = 1000
d = int(input("Please enter the distance of two moon"))
point = moon(num, d, 10, 6)
X_1, X_2, Y_1, Y_2 = point[0], point[1], point[2], point[3]
zeros = np.ones(num)
ones = zeros -2
true = np.concatenate((zeros, ones))
x_1= X_1+X_2
x_1=np.reshape(x_1,(-1,1))
x_2=Y_1+Y_2
x_2=np.reshape(x_2,(-1,1))
true=np.reshape(true,(-1,1))
x_0=np.ones(len(x_1))
x_0=x_0.reshape(-1,1)
x_all=np.concatenate([x_0,x_1,x_2],axis=1)

# Implement regularized least squares (RLS) solution
# λ=0.1
e = 0.1*np.eye(3)
# Calculate the weight [w0,w1,w2]
w=np.linalg.inv((x_all.T.dot(x_all))+e).dot(x_all.T.dot(true))
w=np.reshape(w,(1,3))

# Create dataset for testing
dataTest = getDataSet(2000, d, 10, 6)

# Draw all the points in testing dataset
p = []
true2=[]
for x in dataTest:
    plt.figure(1,figsize=(10,8))
    predict = sgnFunc(x, w[0])
    if predict == 1:
        plt.plot(x[0], x[1], marker='x', color='r',label="bl")
    else:
        plt.plot(x[0], x[1], marker='o', color='b',label='rd')
    true2.append(x[2])
    p.append(predict)

# Caculate mean squared error
mse_validation = sum([(i - j) ** 2 for i, j in zip(p,true2)])/ len(true2)
print("MSE:",mse_validation)

# Draw decision boundary
plt.figure(1)
draw_line(w[0], dataTest)
plt.axis([-20, 30, -20, 20])
plt.show()