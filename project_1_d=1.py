# CSC/ECE/DA 427/527
# Fall 2020

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import mean_squared_error


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


class RBPerceptron:

    def __init__(self, number_of_epochs, learning_rate):
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate

    # Train perceptron
    def train(self, X, D):
        # Initialize weights vector with zeroes
        num_features = X.shape[1]
        self.w = np.zeros(num_features + 1)
        rate=self.learning_rate
        # Perform the epochs
        MSE = []

        for i in range(self.number_of_epochs):
            true = []
            pre = []
            # For every combination of (X_i, D_i)
            for sample, desired_outcome in zip(X, D):
                # Generate prediction and compare with desired outcome
                prediction = self.predict(sample)
                difference = (desired_outcome - prediction)
                true.append(desired_outcome)
                pre.append(prediction)
                # Compute weight update via Perceptron Learning Rule
                if difference != 0:
                    weight_update = rate * difference
                    self.w[1:] += weight_update * sample
                    self.w[0] += weight_update
            m = mean_squared_error(true, pre)
            MSE.append([i + 1, m])
            rate= rate - 0.0019998

        return self, MSE

    def test(self,test):
        result=[]
        num_features = test.shape[1]
        for sample in test:
            outcome = self.predict(sample)
            result.append(outcome)
        return result


    # Generate prediction
    def predict(self, sample):
        outcome = np.dot(sample, self.w[1:]) + self.w[0]
        return np.where(outcome > 0, 1, 0)

num=1000
point = moon(num, 1, 10, 6)
testpoint = moon(num, 1, 10, 6)
X_1, X_2, Y_1, Y_2 = point[0], point[1], point[2], point[3]
X_3, X_4, Y_3, Y_4 = testpoint[0], testpoint[1], testpoint[2], testpoint[3]
small = []
large = []
test=[]
for i in range(len(X_1)):
    p = [X_1[i], Y_1[i], 0]
    small.append(p)
    p2 = [X_2[i], Y_2[i], 1]
    large.append(p2)
for i in range(len(X_3)):
    p = [X_3[i], Y_3[i]]
    test.append(p)
for i in range(len(X_4)):
    p2 = [X_4[i], Y_4[i]]
    test.append(p2)

# Generate target classes {0, 1}
zeros = np.zeros(num)
ones = zeros + 1
true = np.concatenate((zeros, ones))

X_s = np.concatenate((small, large))
X = []
targets = []
np.random.shuffle(X_s)
test = np.array(test)


for i in range(len(X_s)):
    x = X_s[i][0]
    y = X_s[i][1]
    t = X_s[i][2]
    X.append([x, y])
    targets.append(t)

X = np.array(X)
D = targets
D = np.array(D)


rbp = RBPerceptron(50, 0.1)
trained_model, MSE= rbp.train(X, D)
prediction = rbp.test(test)
prediction = np.array(prediction)
error=0
for i in range(len(prediction)):
    if prediction[i] != true[i]:
        error+=1
error_rate = (error/(2*num))*100
print("The error rate is:",error_rate,"%")

# plot_decision_regions(X, D.astype(np.int64), clf=trained_model)
# plt.title('Perceptron')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.figure()

plot_decision_regions(test, prediction.astype(np.int64), clf=trained_model)
plt.title('Testing Result')
plt.xlabel('x')
plt.ylabel('y')
plt.figure()

MSE = np.array(MSE)
plt.plot(MSE[:, 0], MSE[:, 1], color='black')
plt.title('Learning Curve')
plt.xlabel('Number of epochs')
plt.ylabel('MSE')
plt.show()

