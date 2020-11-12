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
def draw_line(w, data,b):

    x = np.linspace(np.amin(data),np.amax(data),100)
    y = -(b+x*w[0])/w[1]
    plt.plot(x, y, '--k',label="DB")

# Get data points
def getDataSet(num_points, distance, radius, width):
    x1, x2, y1, y2 = moon(num_points, distance, radius, width)
    data = []
    data.extend([x1[i], y1[i],1] for i in range(num_points))
    data.extend([x2[i], y2[i],-1] for i in range(num_points))
    return data




a = 0.001
expect_e = 0.05 # Expected Error
maxtrycount = 500

# Activation Function
def sgn(v):
  if v > 0:
    return 1
  else:
    return -1

def get_v(w, x,b):
  return sgn (b+np.dot(w.T, x))

def get_error(w, x, d,b):
  return d  - (b +np.dot(w.T, x))

# Update Weights
'''
  W (n + 1) = W (n) + a * X (n) * e
'''
def getNewWeight(oldW, d, x, a,b):
    e = get_error (w, x, d,b)

    return (oldW + a * x * e, e)

def getbias(bias,e,a):
    b=bias+a*e
    return b

# Create dataset for training
num = 1000
d = int(input("Please enter the distance of two moon"))


#Initial the  w as [0, 0]
# w = [0 for _ in range(2)]
w= np.array([0,0])

# bias =0.1
bias=0.1

#Initial learningRate as 0.001
learningRate = 0.001


#Train the model
dataTrain = getDataSet(num, d, 10, 6)
dataTrain = np.array (dataTrain)

# Training
count = 0
M=[]
while True:

    np.random.shuffle(dataTrain)
    X = []
    true = []
    for i in range(len(dataTrain)):
        x = dataTrain[i][0]
        y = dataTrain[i][1]
        t = dataTrain[i][2]
        X.append([x, y])
        true.append(t)
    X = np.array(X)
    true = np.array(true)

    error = 0
    i = 0
    for xn in X:
        w, e = getNewWeight (w, true[i], xn, learningRate,bias)
        bias = getbias (bias,e,a)
        i += 1
        error += pow(e, 2) #calculate MES
    error = error/float(i)
    count += 1
    M.append([count,error])
    if error < expect_e or count >= maxtrycount:
        break

print ("Final Weight: ", bias,w.T)

# Create dataset for testing
dataTest = getDataSet(2000, d, 10, 6)
dataTest = np.array(dataTest)
X=[]
true2=[]
for i in range(len(dataTest)):
    x = dataTest[i][0]
    y = dataTest[i][1]
    t = dataTest[i][2]
    X.append([x, y])
    true2.append(t)

dataTest = np.array(X)
true2 = np.array(true2)
# Draw all the points in testing dataset
p = []
for x in dataTest:
    plt.figure(1,figsize=(10,8))
    predict = get_v(w, x,bias)
    p.append(predict)
    if predict == 1:
        plt.plot(x[0], x[1], marker='x', color='r',label="bl")
    else:
        plt.plot(x[0], x[1], marker='o', color='b',label='rd')


# Caculate mean squared error
mse_validation = sum([(i - j) ** 2 for i, j in zip(p,true2)])/ len(true2)
print("MSE:",mse_validation)

# Draw the decision boundary
plt.figure(1)
draw_line(w.T, dataTest,bias)
plt.axis([-20, 30, -20, 20])
plt.figure()

M = np.array(M)
plt.plot(M[:, 0], M[:, 1], color='black')
plt.title('Learning Curve')
plt.xlabel('Number of epochs')
plt.ylabel('MSE')
plt.show()