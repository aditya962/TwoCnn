import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat
import random
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def loadLabel(path):
    '''
    :param path:
    :return: training sample label, test sample label
    '''
    assert os.path.exists(path), '{},路径不存在'.format(path)
    # keys:{train_gt, test_gt}
    gt = loadmat(path)
    return gt['train_gt'], gt['test_gt']


def splitSampleByClass(gt, ratio, seed=971104):
    '''
    :param gt: sample label
    :param ratio: Proportion of samples of each type randomly sampled
    :param seed: random seed
    :return: training samples, test samples
    '''
    # Set random seed
    random.seed(seed)
    train_gt = np.zeros_like(gt)
    test_gt = np.copy(gt)
    train_indices = []
    nc = int(np.max(gt))
    # Start randomly selecting samples
    for c in range(1, nc + 1):
        samples = np.nonzero(gt == c)
        sample_indices = list(zip(*samples))
        size = int(len(sample_indices) * ratio)
        x = random.sample(sample_indices, size)
        train_indices += x
    indices = tuple(zip(*train_indices))
    train_gt[indices] = gt[indices]
    test_gt[indices] = 0
    return train_gt, test_gt

# Remove noise bands
def denoise(datasetName, data):
    if datasetName == 'Salinas':
        h, w, _ = data.shape
        x = np.zeros((h, w, 200))
        x[..., 0:106] = data[..., 0:106]
        x[..., 106:144] = data[..., 108:146]
        x[..., 144:] = data[..., 148:]
    elif datasetName == 'Pavia':
        h, w, _ = data.shape
        x = np.zeros((h, w, 100))
        x[:] = data[..., 2:]
    elif datasetName == 'PaviaU':
        h, w, _ = data.shape
        x = np.zeros((h, w, 100))
        x[:] = data[..., 3:]
    elif datasetName == 'Indian':
        h, w, _ = data.shape
        x = np.zeros((h, w, 176))
        data =data.reshape((h*w, -1))
        variance = np.std(data, axis=0)
        index = np.argsort(variance)
        index = index[24:]
        index = np.sort(index)
        data = data[:, index]
        x[:] = data.reshape((h, w, -1))
    else:
        x = data
    return x

