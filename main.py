import numpy as np
import gzip
import os
import urllib.request
from Network import Network
# 1. Hàm tải dữ liệu (nếu chưa có)
def download_mnist(save_dir='./data'):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file in files:
        path = os.path.join(save_dir, file)
        if not os.path.exists(path):
            print(f"Đang tải {file}...")
            urllib.request.urlretrieve(base_url + file, path)
    print("Đã tải xong dữ liệu!")

# 2. Hàm đọc file Images
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Bỏ qua 16 byte đầu (header info: magic number, number of images, rows, cols)
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # Reshape thành (số lượng ảnh, 784) - Flatten luôn để đưa vào mạng NN
    return data.reshape(-1, 784)

# 3. Hàm đọc file Labels
def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Bỏ qua 8 byte đầu (header info)
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# --- CHẠY ---
download_mnist()

X_train = load_images('./data/train-images-idx3-ubyte.gz')
y_train = load_labels('./data/train-labels-idx1-ubyte.gz')
X_test = load_images('./data/t10k-images-idx3-ubyte.gz')
y_test = load_labels('./data/t10k-labels-idx1-ubyte.gz')

print(f"X_train shape: {X_train.shape}") # (60000, 784)
print(f"y_train shape: {y_train.shape}") # (60000,)


onehot_matrix = np.zeros((y_train.size, 10))
onehot_matrix[np.arange(y_train.size), y_train] = 1
y_train = onehot_matrix

# mean = np.mean(X_train)
# std = np.std(X_train)
# X_train = (X_train-mean) / (std + 1e-8)
#
# mean = np.mean(X_train, axis=1, keepdims=True) # Shape (60000, 1)
# std = np.std(X_train, axis=1, keepdims=True)
# X_train = (X_train - mean) / (std + 1e-8)

# Tính dọc theo trục 0 (dọc theo các mẫu dữ liệu)
mean = np.mean(X_train, axis=0) # Shape ra (784,)
std = np.std(X_train, axis=0)   # Shape ra (784,)

# Cộng epsilon để tránh chia cho 0 ở những pixel luôn luôn đen (std=0)
X_train = (X_train - mean) / (std + 1e-8)


network = Network([784,512,10], batch_size=128, lr=0.1, epochs=1000)
network.train(X_train, y_train)