
from model import FeedforwardNeuralNetwork
from utils import relu, relu_prime, sigmoid, sigmoid_prime

def build_model(input_dim, hidden_units, seed=42): 
    return FeedforwardNeuralNetwork(
        num_features=input_dim,
        num_hidden_units=list(hidden_units),
        num_classes=1,
        activation_func=[relu, relu, sigmoid],
        activation_func_prime=[relu_prime, relu_prime, sigmoid_prime],
        weights=None,
    )