import config
import numpy as np

class Neural_Network(object):

    def __init__(self, input, output, hidden):
        #Define hyperparameters
        self.inputLayerSize = input
        self.outputLayerSize = output
        self.hiddenLayerSize = hidden

        #Initial Weights - set to random values
        self.W1 = config.initW1
        self.W2 = config.initW2

    def forward(self, X):
        # Propagate forward
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
