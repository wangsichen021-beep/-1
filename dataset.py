import numpy as np
import urllib.request
import gzip
import os

def load_fashion_mnist(path='./data'):
    """下载并加载 Fashion-MNIST 数据集"""
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = [
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
    ]
    os.makedirs(path, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file_path)

    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28) / 255.0 # 归一化到 0-1

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train_full = read_images(os.path.join(path, files[0]))
    y_train_full = read_labels(os.path.join(path, files[1]))
    X_test = read_images(os.path.join(path, files[2]))
    y_test = read_labels(os.path.join(path, files[3]))

    # 划分验证集 (后10000张作为验证集)
    X_train, X_val = X_train_full[:-10000], X_train_full[-10000:]
    y_train, y_val = y_train_full[:-10000], y_train_full[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test