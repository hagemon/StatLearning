from torchvision.datasets import FashionMNIST
import numpy as np
import os
from sklearn.decomposition import PCA


def process_data():
    mnist_train = FashionMNIST('data', train=True, download=True)
    data = mnist_train.data.numpy()
    label = mnist_train.targets.numpy()
    perm = np.array([i for i in range(len(data))])
    np.random.shuffle(perm)
    pca = PCA(32)
    train_data = data[perm[:100]].reshape(100, 784)
    train_label = label[perm[:100]]
    train_data = pca.fit_transform(train_data)
    test_data = data[perm[100:150]].reshape(50, 784)
    test_label = data[perm[100:150]]
    test_data = pca.fit_transform(test_data)
    np.save('data/data.npy', {
        'train_data': train_data,
        'train_label': train_label,
        'test_data': test_data,
        'test_label': test_label
    })


def get_data():
    path = 'data/data.npy'
    if not os.path.exists(path):
        process_data()
    data = np.load(path).item()
    return data['train_data'], data['train_label'], data['test_data'], data['test_label']
