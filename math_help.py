import numpy as np


def sigmoid(z):
    # sigmoid function -> high value = 1, low value = 0
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1 - sig)
