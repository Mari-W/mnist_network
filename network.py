import math_help as mh
import numpy as np
import random as rnd
import _pickle as pkl


class Network(object):
    def __init__(self, sizes):
        # count of layers
        self.num_layers = len(sizes)
        # vector of layer sizes
        self.sizes = sizes
        # biases or "threshold" -> how easy is sigmoid neuron activated (not for first  layer)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # weights -> how strong does the connection from n-1 layer neuron affect the n neuron
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        # for every layer it takes the bias vector and the weights corresponding to the layer before
        for b, w in zip(self.biases, self.weights):
            # calculates the activation of itself by adding the bias to the dot product of weights and activation
            # e.g input values
            a = mh.sigmoid(np.dot(w, a) + b)
        # return activation vector of output layer
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None, save=False):
        # amount of training/test data provided
        if training_data:
            n_test = len(test_data)
        n = len(training_data)
        # for every epoch (running through all mini batches and furthermore the whole data set
        for j in range(epochs):
            # shuffle training data on begin of ever epoch and split it into mini badges
            rnd.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            # loop through mini badges and update them accordingly to back prop calculation
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # test if data if test_data is provided and evaluates how good network currently is
            # by using data not included in training data
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            if save:
                pkl.dump(self, open("network.obj", 'wb'))

    def update_mini_batch(self, mini_batch, eta):
        # create empty vectors for "wanted change" for whole mini batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # for data and result to data
        for x, y in mini_batch:
            # calculate back propagation for every training example
            delta_nabla_b, delta_nabla_w = self.back_prop(x, y)
            # add "wanted change" to vector of mini batch
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update weights and biases according to "wanted change" in respect of the learning rate
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def back_prop(self, x, y):
        # create empty vectors for "wanted change" for one single training example
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # activation of current layer, for first layer is inputs
        activation = x
        # store activations of all layers
        activations = [x]
        # store z= dot(w, a)+b of all layers
        zs = []
        # feed forward like in feed forward method but saving all as and  zs, not only returning the output layer
        for b, w in zip(self.biases, self.weights):
            # compute z (non-sigmoid activation of next layer)
            z = np.dot(w, activation) + b
            zs.append(z)
            # compute a
            activation = mh.sigmoid(z)
            activations.append(activation)
        # back prop
        # calculate error for given input in output layer only and assign cost derivative (error = cost*sigmoid_prime(z)
        delta = (activations[-1] - y) * mh.sigmoid_prime(zs[-1])
        # cost derivative for the biases is just the error
        # -> there is one error per neuron per layer and one bias per neuron per layer
        nabla_b[-1] = delta
        # cost derivatives for weights in respect of the activation from the layer before
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # calculate error for the layers before the last layer (going through backwards of cause)
        for layer in range(2, self.num_layers):
            # z from layer
            z = zs[-layer]
            # error in respect of layer before (delta) and the weights connecting to this layer
            # (error=weights_connecting_to_layer*error_of_layer_before) each multiplied with sigmoid prime of z
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * mh.sigmoid_prime(z)
            # again cost derivative for biases is just delta
            nabla_b[-layer] = delta
            # again cost derivatives for weights in respect of the activation from the layer before
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        # feeds test data through network and takes the highest output value as result
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        # sums up all right results by comparing result of network x with right output y
        return sum(int(x == y) for (x, y) in test_results)
