import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA


class HSIDataset(Dataset):
    def __init__(self, data, label, patchsz=1):
        '''
        :param data: [h, w, bands]
        :param label: [h, w]
        :param n_components: scale
        :param patchsz: scale
        '''
        super(HSIDataset, self).__init__()
        self.data = data # [h, w, bands]
        self.label = label # [h, w]
        self.patchsz = patchsz
        # Dimensions of original data
        self.h, self.w, self.bands = self.data.shape
        self.Normalize()
        # self.get_mean()
        # # Data centralization
        # self.data -= self.mean
        self.addMirror()

    # Data normalization
    def Normalize(self):
        data = self.data.reshape((self.h * self.w, self.bands))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        self.data = data.reshape((self.h, self.w, self.bands))

    # Add image
    def addMirror(self):
        dx = self.patchsz // 2
        if dx != 0:
            mirror = np.zeros((self.h + 2 * dx, self.w + 2 * dx, self.bands))
            mirror[dx:-dx, dx:-dx, :] = self.data
            for i in range(dx):
                # Fill the upper left part of the image
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # Fill in the lower right part of the image
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
            self.data = mirror

    def __len__(self):
        return self.h * self. w

    def __getitem__(self, index):
        '''
        :param index:
        :return: Element spectral information, element spatial information, label
        '''
        l = index // self.w
        c = index % self.w
        # field: [patchsz, patchsz, bands]
        neighbor_region = self.data[l:l + self.patchsz, c:c + self.patchsz, :]
        # Take the mean
        # neighbor_region_mean = neighbor_region
        neighbor_region_mean = np.mean(neighbor_region, axis=-1, keepdims=True)
        # Spectrum of center pixel
        spectra = self.data[l + self.patchsz // 2, c + self.patchsz // 2]
        # category
        target = self.label[l, c] - 1
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region_mean, dtype=torch.float32)), \
        torch.tensor(target, dtype=torch.long)


class HSIDatasetV1(HSIDataset):
    def __init__(self, data, label, patchsz=1):
        super().__init__(data, label, patchsz)
        self.sampleIndex = list(zip(*np.nonzero(self.label)))

    def __len__(self):
        return len(self.sampleIndex)

    def __getitem__(self, index):
        l, c = self.sampleIndex[index]
        spectra = self.data[l + self.patchsz // 2, c + self.patchsz // 2]
        neighbor_region = self.data[l:l + self.patchsz, c:c + self.patchsz, :]
        # Tags are encoded starting from 0
        target = self.label[l, c] - 1
        # Take the mean
        neighbor_region_mean = np.mean(neighbor_region, axis=-1, keepdims=True)
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region_mean, dtype=torch.float32)), \
                torch.tensor(target, dtype=torch.long)

class DatasetInfo(object):
    info = {'PaviaU': {
        'data_key': 'paviaU',
        'label_key': 'paviaU_gt'
    },
        'Salinas': {
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt'
        },
        'KSC': {
            'data_key': 'KSC',
            'label_key': 'KSC_gt'
    },  'Houston':{
            'data_key': 'Houston',
            'label_key': 'Houston2018_gt'
    },  'Indian':{
            'data_key': 'indian_pines_corrected',
            'label_key': 'indian_pines_gt'
    },  'Pavia':{
            'data_key': 'pavia',
            'label_key': 'pavia_gt'
    }}


# from scipy.io import loadmat
# import numpy as np
# m = loadmat('data/PaviaU/PaviaU.mat')
# data = m['paviaU']
# m = loadmat('data/PaviaU/PaviaU_gt.mat')
# label = m['paviaU_gt']
# data, label = data.astype(np.float32), label.astype(np.long)
# dataset = HSIDataset(data, label, patchsz=21)
# w = data.shape[1]
# index = 150
# l, c = index // w, index % w
# (spectra, neighbor_region), target = dataset[index]
# print(torch.equal(spectra, neighbor_region[21 // 2, 21 // 2]))
# print(target - 1)
# print(label[l, c])
