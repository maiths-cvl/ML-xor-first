import numpy as np
import time

td = time.time()

np.random.seed(0)

µ = 0.1 #taux d'apprentissage

def ReLU(x):
    return np.maximum(0, x)

def DReLU(x):
    return (x > 0).astype(float)

def Sig(x):
    return (1/(1+np.exp(-x)))

def DSig(x):
    s = Sig(x)
    return s * (1-s)

class Layer:
    def __init__(self, n_input, n_output, couche):
        self.couche = couche
        self.weights = np.random.randn(n_output, n_input)
        self.biases = np.random.randn(1, n_output) * 0.1

    def forward(self, input):
        self.input = np.array(input)
        if self.input.ndim == 1:
            self.input = self.input.reshape(1, -1)
        self.z = self.input @ self.weights.T + self.biases # input : (n*2) x (1*2)^T = (n*2) x (2*1) => (n*1)
        if self.couche == 1:
            self.output = ReLU(self.z)
        if self.couche == 2:
            self.output = Sig(self.z)

    def backward(self, loss):
        if self.couche == 1:
            self.delta = loss * DReLU(self.z) # delta : output (n*1) * z (2*1) x (1*2) = (n*1)x(2*2) = (n*2) 
        if self.couche == 2:
            self.delta = loss * DSig(self.z)
        self.dw = self.delta.T @ self.input # inputs : (n*2) donc (n*2) x (n*2) => pb dimensions => (n*2) x (n*2)^T mieux qui est fait
        self.db = np.sum(self.delta, axis=0, keepdims=True)

        self.weights -= µ * self.dw
        self.biases -= µ * self.db

        self.back = self.delta @ self.weights
    
def error(x, y, d=False):
    if d == True:
        l = 2*(x-y)
    else :
        l = (x-y)**2
    return l

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

Y = [[0],
     [1],
     [1],
     [0]]

layer1 = Layer(2, 2, 1)  
layer1.forward(X)
layer2 = Layer(2, 1, 2)
layer2.forward(layer1.output)
layer2.backward(error(layer2.output, Y, d=True))
layer1.backward(layer2.back)

precision = 99.9 # en pourcentage

while np.mean(error(layer2.output, Y))>(1-precision)/100**2:
    layer1.forward(X)
    layer2.forward(layer1.output)

    layer2.backward(error(layer2.output, Y, d=True))
    layer1.backward(layer2.back)

layer1.forward(X)
layer2.forward(layer1.output)

tf = time.time()
print("took", tf-td,"s to compute")

for i in layer2.output:
    print(i*100, "%")

print(layer1.weights, layer1.biases, layer2.weights, layer2.biases)