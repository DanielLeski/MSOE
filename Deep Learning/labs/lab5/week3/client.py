import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import warnings
import os.path

# warnings.filterwarnings('ignore')  # If you see warnings that you know you can ignore, it can be useful to enable this.

EPOCHS = 40
# For simple regression problem
TRAINING_POINTS = 1000

# For fashion-MNIST and similar problems
DATA_ROOT = '/data/cs3450/data/'
FASHION_MNIST_TRAINING = '/data/cs3450/data/fashion_mnist_flattened_training.npz'
FASHION_MNIST_TESTING = '/data/cs3450/data/fashion_mnist_flattened_testing.npz'
CIFAR10_TRAINING = '/data/cs3450/data/cifar10_flattened_training.npz'
CIFAR10_TESTING = '/data/cs3450/data/cifar10_flattened_testing.npz'
CIFAR100_TRAINING = '/data/cs3450/data/cifar100_flattened_training.npz'
CIFAR100_TESTING = '/data/cs3450/data/cifar100_flattened_testing.npz'

# With this block, we don't need to set device=DEVICE for every tensor.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.cuda.set_device(0)
     torch.set_default_tensor_type(torch.cuda.FloatTensor)
     print("Running on the GPU")
else:
     print("Running on the CPU")

def create_linear_training_data():
    """
    This method simply rotates points in a 2D space.
    Be sure to use L2 regression in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_folded_training_data():
    """
    This method introduces a single non-linear fold into the sort of data created by create_linear_training_data. Be sure to REMOVE the final softmax layer before testing on this data!
    Be sure to use L2 regression in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    x2 *= 2 * ((x2 > 0).float() - 0.5)
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_square():
    """
    This is a square example
    insideness is true if the points are inside the square.
    :return: (points, insideness) the dataset. points is a 2xN array of points and insideness is true if the point is inside the square.
    """
    win_x = [2,2,3,3]
    win_y = [1,2,2,1]
    win = torch.tensor([win_x,win_y],dtype=torch.float32)
    win_rot = torch.cat((win[:,1:],win[:,0:1]),axis=1)
    t = win_rot - win # edges tangent along side of poly
    rotation = torch.tensor([[0, 1],[-1,0]],dtype=torch.float32)
    normal = rotation @ t # normal vectors to each side of poly
        # torch.matmul(rotation,t) # Same thing

    points = torch.rand((2,2000),dtype = torch.float32)
    points = 4*points

    vectors = points[:,np.newaxis,:] - win[:,:,np.newaxis] # reshape to fill origin
    insideness = (normal[:,:,np.newaxis] * vectors).sum(axis=0)
    insideness = insideness.T
    insideness = insideness > 0
    insideness = insideness.all(axis=1)
    return points, insideness


def create_patterns():
    """
    I don't remember what sort of data this generates -- Dr. Yoder

    :return: (points, insideness) the dataset. points is a 2xN array of points and insideness is true if the point is inside the square.
    """
    pattern1 = torch.tensor([[1, 0, 1, 0, 1, 0]],dtype=torch.float32).T
    pattern2 = torch.tensor([[1, 1, 1, 0, 0, 0]],dtype=torch.float32).T
    num_samples = 1000

    x = torch.zeros((pattern1.shape[0],num_samples))
    y = torch.zeros((2,num_samples))
    # TODO: Implement with shuffling instead?
    for i in range(0,num_samples):
        if torch.rand(1) > 0.5:
            x[:,i:i+1] = pattern1
            y[:,i:i+1] = torch.tensor([[0,1]],dtype=torch.float32).T
        else:
            x[:,i:i+1] = pattern2
            y[:,i:i+1] = torch.tensor([[1,0]],dtype=torch.float32).T
    return x, y


def load_dataset_flattened(train=True,dataset='Fashion-MNIST',download=False):
    """
    :param train: True for training, False for testing
    :param dataset: 'Fashion-MNIST', 'CIFAR-10', or 'CIFAR-100'
    :param download: True to download. Keep to false afterwords to avoid unneeded downloads.
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    if dataset == 'Fashion-MNIST':
        if train:
            path = FASHION_MNIST_TRAINING
        else:
            path = FASHION_MNIST_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-10':
        if train:
            path = CIFAR10_TRAINING
        else:
            path = CIFAR10_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-100':
        if train:
            path = CIFAR100_TRAINING
        else:
            path = CIFAR100_TESTING
        num_labels = 100
    else:
        raise ValueError('Unknown dataset: '+str(dataset))

    if os.path.isfile(path):
        print('Loading cached flattened data for',dataset,'training' if train else 'testing')
        data = np.load(path)
        x = torch.tensor(data['x'],dtype=torch.float32)
        y = torch.tensor(data['y'],dtype=torch.float32)
        pass
    else:
        class ToTorch(object):
            """Like ToTensor, only redefined by us for 'historical reasons'"""

            def __call__(self, pic):
                return torchvision.transforms.functional.to_tensor(pic)

        if dataset == 'Fashion-MNIST':
            data = torchvision.datasets.FashionMNIST(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-10':
            data = torchvision.datasets.CIFAR10(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-100':
            data = torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        else:
            raise ValueError('This code should be unreachable because of a previous check.')
        x = torch.zeros((len(data[0][0].flatten()), len(data)),dtype=torch.float32)
        for index, image in enumerate(data):
            x[:, index] = data[index][0].flatten()
        labels = torch.tensor([sample[1] for sample in data])
        y = torch.zeros((num_labels, len(labels)), dtype=torch.float32)
        y[labels, torch.arange(len(labels))] = 1
        np.savez(path, x=x.numpy(), y=y.numpy())
    return x, y


import network
import layers
from network import * 
from layers import *

# +
#Client By Hand Example
net = Network()
x = torch.tensor([6,7]).reshape(2,1)
w = torch.tensor([[0,1,2], [3,4,5]]).reshape(3,2)
b = torch.tensor([1,3,5]).reshape(3,1)

print(x.shape)
print(w.shape)
print(b.shape)
#setting the features as input
features = Input((x.shape))
features.set(x)

weights = Input((w.shape))
weights.set(w)

bias = Input((b.shape))
bias.set(b)

first_linear_layer = Linear(features, weights, bias)

#first_linear_layer.forward()
#print(first_linear_layer.output)

relu = ReLU(first_linear_layer)
#relu.forward()
#print(relu.output)

#net.add(features)
#net.add(weights)
#net.add(bias)
net.add(first_linear_layer)
net.add(relu)
net.set_input(features)
net.set_output(relu)

net.forward()
# -


