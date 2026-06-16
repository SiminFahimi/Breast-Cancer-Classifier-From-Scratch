
from model import FeedforwardNeuralNetwork
from utils import relu, relu_prime, sigmoid, sigmoid_prime
import numpy as np

def build_model(input_dim, seed=42):
    np.random.seed(seed)
    return FeedforwardNeuralNetwork(
        num_features=input_dim,
        num_hidden_units=[16, 8],
        num_classes=1,
        activation_func=[relu, relu, sigmoid],
        activation_func_prime=[relu_prime, relu_prime, sigmoid_prime],
        weights=None,
    )