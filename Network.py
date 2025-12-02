from typing import Optional

import numpy as np
import random
from tqdm import tqdm
def sigmoid(z: np.ndarray):
    return 1.0/(1.0 + np.exp(-z))

class Network:
    def __init__(self, sizes: list[int], batch_size: int, lr: float, epochs: int):
        self.sizes = sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) * 0.01 for x,y in zip(sizes[:-1], sizes[1:])]
        print(len(self.weights))
        print(self.biases[1].shape)

    def forward(self, x: np.ndarray):
        new_z = x.copy().transpose()
        z_list = [new_z.copy()]
        active_list = [new_z.copy()]
        i=0
        for weight, bias in zip(self.weights, self.biases):
  
            new_z = np.dot(weight,new_z) + bias
            z_list.append(new_z)
            active_list.append(sigmoid(new_z))
            i+=1

        return z_list, active_list


    def backward(self, y:np.ndarray, z_list:list[np.ndarray], active_list:list[np.ndarray]):
        weight_grads = []
        bias_grads = []
        last_active = active_list[-1]
        delta = (last_active - y.transpose()) * last_active * (1 - last_active)
        # print("delta: ",delta.shape)
        weight_grads.append(np.dot(delta, np.transpose(active_list[-2])))
        bias_grads.append(np.sum(delta, axis=1, keepdims=True))


        for i in range(self.num_layers - 2, 0, -1):
            next_w = self.weights[i]
            a = active_list[i]
            a_prev = active_list[i-1]
            delta = np.dot(np.transpose(next_w), delta) * a * (1-a)
            weight_grads.append(np.dot(delta, np.transpose(a_prev)))
            bias_grads.append(np.sum(delta, axis=1, keepdims=True))

        return list(reversed(weight_grads)), list(reversed(bias_grads))

    def updates(self, weight_grads: list[np.ndarray], bias_grads: list[np.ndarray], step: int):
        for i in range(0, self.num_layers - 1):

            self.weights[i] = self.weights[i] - self.lr * weight_grads[i]/step
            self.biases[i] = self.biases[i] - self.lr * bias_grads[i]/step


    def evaluate(self, x_data: np.ndarray, y_data: np.ndarray):

        _, active_list = self.forward(x_data)
        
        output_activations = active_list[-1]
        

        predictions = np.argmax(output_activations, axis=0) 
        

        if y_data.ndim == 2 and y_data.shape[1] > 1:
            y_indices = np.argmax(y_data, axis=1)
        else:
            y_indices = y_data.flatten()

        accuracy = np.sum(predictions == y_indices) / len(y_data)
        
        return accuracy

    def train(self, x: np.ndarray, y: np.ndarray):
        for epoch in range(self.epochs):

            total_loss = 0.0
            num_of_step = x.shape[0] // self.batch_size if x.shape[0] % self.batch_size == 0 else x.shape[0] // self.batch_size + 1
            batch_iters = tqdm(range(0,len(x), self.batch_size), desc=f"Epoch:{epoch + 1}/{self.epochs}", total=num_of_step)
            for i in batch_iters:
                step = min(self.batch_size, len(x) - i)
                x_batch = x[i:i+step]
                y_batch = y[i:i+step]
                z_list, active_list = self.forward(x_batch)

                weight_grads, bias_grads = self.backward(y_batch, z_list, active_list)
                batch_loss = 0.5 * np.sum((active_list[-1] - y_batch.transpose())**2)/step
                total_loss += batch_loss*step
                batch_iters.set_postfix(batch_loss=f"{batch_loss:.4f}", refresh=False)
                self.updates(weight_grads, bias_grads,step)
            avg_loss = total_loss/x.shape[0]
            train_accuracy = self.evaluate(x, y)
            print(f"Epoch {epoch + 1}/{self.epochs} - Avg Loss: {avg_loss:.4f} - Accuracy: {train_accuracy:.4f}")
