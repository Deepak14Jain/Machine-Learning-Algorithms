import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X) #maximum of X array longitudinally
y = y/100

class NeuralNetwork(object):
    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.hiddenLayer_size = 3

        self.W1 = np.random.randn(self.input_size, self.hiddenLayer_size)
        self.W2 = np.random.randn(self.hiddenLayer_size, self.output_size)

    def sigmoid(self, s, deriv = False):
        if deriv == True:
            return s*(1-s)
        return 1/(1+np.exp(-s))

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        self.output = self.sigmoid(self.z3)
        return self.output

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        self.z2_error = np.dot(self.output_delta,self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)

        self.W1 += np.dot(X.T, self.z2_delta)
        self.W2 += np.dot(self.z2.T, self.output_delta)

    def test(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)

NN = NeuralNetwork()
for i in range(100000):
    NN.test(X, y)
print("Predicted Output: ",NN.forward(X))
