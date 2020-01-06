from network import Network
from emnist import extract_training_samples
from emnist import extract_test_samples
import numpy as np
import string


class MNISTNetwork:

    def __init__(self):
        d_inputs, d_results = extract_training_samples("digits")
        l_inputs, l_results = extract_training_samples("letters")
        dt_inputs, dt_results = extract_test_samples("digits")
        lt_inputs, lt_results = extract_test_samples("letters")
        self.training_data = zip(d_inputs + l_inputs, d_results + l_results)
        self.test_data = zip(dt_inputs + lt_inputs, dt_results + lt_results)
        self.outputs = ["0123456789" + string.ascii_lowercase + string.ascii_uppercase]
        self.network = Network([784, 50, 50, len(self.outputs)])
        self.network.sgd(self.training_data, 30, 20, 3.0, self.test_data, True)

    def vectorized_results(self, x):
        # returns vector filled with 0 but a 1 at the correct answers position in the output layer
        e = np.zeros(len(self.outputs), 1)
        e[self.outputs.index(x)] = 1.0
        return e


MNISTNetwork()
