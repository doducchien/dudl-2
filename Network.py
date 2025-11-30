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
        z_list = []
        active_list = []
        for weight, bias in zip(self.weights, self.biases):
            new_z = np.dot(weight,new_z) + bias
            z_list.append(new_z)
            active_list.append(sigmoid(new_z))

        return z_list, active_list


    def backward(self, y:np.ndarray, z_list:list[np.ndarray], active_list:list[np.ndarray]):
        weight_grads = []
        bias_grads = []
        last_active = active_list[-1]
        delta = (last_active - y) * last_active * (1 - last_active)
        weight_grads.append(np.dot(delta, np.transpose(active_list[-2])))
        bias_grads.append(delta)

        zaw_list = list(zip(z_list, active_list, self.weights))
        for i in range(len(zaw_list) - 2, -1, -1):
            _,_,next_w = zaw_list[i+1]
            _,a,_ = zaw_list[i]
            _,a_prev,_ = zaw_list[i-1]
            delta = np.dot(np.transpose(next_w), delta) * a * (1-a)
            weight_grads.append(np.dot(delta, np.transpose(a_prev)))
            bias_grads.append(delta)

        return list(reversed(weight_grads)), list(reversed(bias_grads))

    def updates(self, weight_grads: list[np.ndarray], bias_grads: list[np.ndarray]):
        print('weights: ', len(self.weights))
        print('weight_grads: ', len(weight_grads))
        for i in range(0, self.num_layers - 1):
            print('weight: ', self.weights[i].shape)
            print('weight_grad: ', weight_grads[i].shape)
            self.weights[i] = self.weights[i] - self.lr * weight_grads[i].mean(axis=1)
            self.biases[i] = self.biases[i] - self.lr * bias_grads[i]

    def train(self, x: np.ndarray, y: np.ndarray):
        for i in range(0, len(x), self.batch_size):
            step = min(self.batch_size, len(x) - i)
            z_list, active_list = self.forward(x[i:i+step])
            weight_grads, bias_grads = self.backward(y[i:i+step], z_list, active_list)
            self.updates(weight_grads, bias_grads)
            break


