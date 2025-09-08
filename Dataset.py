import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio

def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data


class ChikuseiDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=16 * 1, patch_w=16 * 1,
                 h_stride=6 * 1, w_stride=8 * 1, ratio=4,
                 type='train'):
        super(ChikuseiDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        if ratio == 4:
            self.rows = 71 * 2
            self.cols = 70 * 2
        elif ratio == 8:
            self.rows = 71 * 1
            self.cols = 70 * 1
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type == 'train':
            self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio,
                                                                          s_h=22, s_w=16)

        if self.type == 'eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=32 * 2, patch_w=32 * 2, ratio=ratio)
        if self.type == 'test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=32 * 2, patch_w=32 * 2, ratio=ratio)

    def getData(self, ratio):
        hrhsi = scio.loadmat(self.mat_save_path + 'Chikusei_HRHSI.mat')['hrhsi']
        lrhsi = scio.loadmat(self.mat_save_path + 'Chikusei_LRHSI{}.mat'.format(ratio))['lrhsi']
        hrmsi = scio.loadmat(self.mat_save_path + 'Chikusei_HRMSI.mat')['hrmsi']
        # Data normalization and scaling[0, 1]
        hrhsi = normalize(hrhsi)
        lrhsi = normalize(lrhsi)
        hrmsi = normalize(hrmsi)

        return hrhsi, lrhsi, hrmsi

    def generateTrain(self, patch_h, patch_w, ratio, s_h, s_w):
        label_patch = np.zeros((s_h * s_w, patch_h * ratio, patch_w * ratio, 128), dtype=np.float32)
        hrmsi_patch = np.zeros((s_h * s_w, patch_h * ratio, patch_w * ratio, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((s_h * s_w, patch_h, patch_w, 128), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[:self.rows * ratio, :, :]
        lrhsi = lrhsi[:self.rows, :, :]
        hrmsi = hrmsi[:self.rows * ratio, :, :]

        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, self.rows - patch_h + 1, self.h_stride):
            for y in range(0, self.cols - patch_w + 1, self.w_stride):
                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 128), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, 128), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[self.rows * ratio:, :patch_w * ratio, :]
        lrhsi = lrhsi[self.rows:, :patch_w, :]
        hrmsi = hrmsi[self.rows * ratio:, :patch_w * ratio, :]

        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 128), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, 128), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[self.rows * ratio:, patch_w * ratio:patch_w * ratio * 2, :]
        lrhsi = lrhsi[self.rows:, patch_w:patch_w * 2, :]
        hrmsi = hrmsi[self.rows * ratio:, patch_w * ratio:patch_w * ratio * 2, :]

        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2, 0, 1))
        hrmsi = np.transpose(self.msi_data[index], (2, 0, 1))
        lrhsi = np.transpose(self.hsi_data[index], (2, 0, 1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]
