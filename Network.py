from typing import Optional

import numpy as np
import random

def sigmoid(z: np.ndarray):
    return 1.0/(1.0 + np.exp(-z))

class Network:
    def __init__(self, sizes: list[int], batch_size: int, lr: float):
        self.sizes = sizes
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        print(len(self.weights))

    def forward(self, x: np.ndarray):
        new_z = x.copy().transpose()
        z_list = [new_z.copy()]
        active_list = [new_z.copy()]
        for weight, bias in zip(self.weights, self.biases):
            new_z = np.dot(weight,new_z) + bias
            z_list.append(new_z)
            active_list.append(sigmoid(new_z))

        return z_list, active_list


    def backward(self, y:np.ndarray, z_list:list[np.ndarray], active_list:list[np.ndarray]):
        weight_grads = []
        bias_grads = []
        last_active = active_list[-1]
        print(y)
        delta = (last_active - y.transpose()) * last_active * (1 - last_active)
        weight_grads.append(np.dot(delta, np.transpose(active_list[-2])))
        bias_grads.append(delta)


        for i in range(self.num_layers - 2, 0, -1):
            print("i: ", i)
            next_w = self.weights[i]
            a = active_list[i]
            a_prev = active_list[i-1]
            delta = np.dot(np.transpose(next_w), delta) * a * (1-a)
            weight_grads.append(np.dot(delta, np.transpose(a_prev)))
            bias_grads.append(delta)

        return list(reversed(weight_grads)), list(reversed(bias_grads))

    def updates(self, weight_grads: list[np.ndarray], bias_grads: list[np.ndarray]):
        for i in range(0, self.num_layers - 1):
            self.weights[i] = self.weights[i] - self.lr * weight_grads[i]
            self.biases[i] = self.biases[i] - self.lr * bias_grads[i]

    def train(self, x: np.ndarray, y: np.ndarray):
        total_loss = 0.0
        for i in range(0, len(x), self.batch_size):
            step = min(self.batch_size, len(x) - i)
            x_batch = x[i:i+step]
            y_batch = y[i:i+step]
            z_list, active_list = self.forward(x_batch)

            weight_grads, bias_grads = self.backward(y_batch, z_list, active_list)
            total_loss += 0.5 * np.sum((active_list[-1] - y_batch)**2)/step
            self.updates(weight_grads, bias_grads)


