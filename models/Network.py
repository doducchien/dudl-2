from typing import Optional

import numpy as np
import random

def sigmoid(z: np.ndarray):
    return 1.0/(1.0 + exp(-z))

class Network:
    def __init__(self, sizes: list[int]):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def forward(self, x: np.ndarray, mode: Optional[str] = "training"):
        if mode == 'training':
            z = x.copy()
            z_list = [z]
            active_z_list = [z]
            for weight, bias in zip(self.weights, self.biases):
                new_z = np.dot(weight,z) + bias
                z_list.append(new_z)
                active_z_list.append(sigmoid(new_z))
            return np.array(z_list), np.array(active_z_list)

        for weight, bias in zip(self.weights, self.biases):
            x = sigmoid(np.dot(weight,x) + bias)
        return x

    def backward(self, x:np.ndarray, y:np.ndarray, z_list:np.ndarray, active_z_list:np.ndarray):
        weight_grads = np.zeros_like((1,sizes))
        bias_grads = []





