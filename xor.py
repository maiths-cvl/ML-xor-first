import numpy as np

np.random.seed(0)

µ = 0.1

def Sig(x):
    return 1 / (1+np.exp(-x))

def DSig(x):
    return Sig(x) * (1-Sig(x))

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_outputs, n_inputs)
        self.biases = np.random.randn(1, n_outputs) * 0.1

    def forward(self, input):
        self.input = np.array(input)
        if self.input.ndim == 1:
            self.input = self.input.reshape(1, -1)

        self.z = input @ self.weights.T + self.biases
        self.output = Sig(self.z)

    def backward(self, err):
        delta = err * DSig(self.z)
        dw = delta.T @ self.input
        db = np.sum(delta, axis=0, keepdims=True)
                    

        self.weights -= µ * dw
        self.biases -= µ * db

        self.back = delta @ self.weights
    
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

Y = [[0],
     [1],
     [1],
     [0]]

layer1 = Layer(2, 2)
layer1.forward(X)
layer2 = Layer(2, 1)
layer2.forward(layer1.output)

def error(x, y, d=False):
    if d==True:
        return 2* (x-y)
    else:
        return (x-y)**2
    
layer2.backward(error(layer2.output, Y, d=True))
layer1.backward(layer2.back)

precision = 99

while np.mean(error(layer2.output, Y))>(1-precision/100)**2:
    layer1.forward(X)
    layer2.forward(layer1.output)

    layer2.backward(error(layer2.output, Y, d=True))
    layer1.backward(layer2.back)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)
print(layer1.weights, layer1.biases, layer2.weights, layer2.biases)

#new