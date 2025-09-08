import torch
import numpy as np
from skimage.metrics import structural_similarity

class fftLoss(torch.nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss

def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)
    return psnr

# def calc_rmse(img_tgt, img_fus):
#     rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))
#
#     return rmse

def calc_rmse(img_tgt, img_fus):

    img_tgt = np.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = np.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

# def calc_sam(img_tgt, img_fus):
    # img_tgt = np.squeeze(img_tgt)
    # img_fus = np.squeeze(img_fus)
    # img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    # img_fus = img_fus.reshape(img_fus.shape[0], -1)
    # img_tgt = img_tgt / np.max(img_tgt)
    # img_fus = img_fus / np.max(img_fus)
    #
    # A = np.sqrt(np.sum(img_tgt**2, axis=0))
    # B = np.sqrt(np.sum(img_fus**2, axis=0))
    # AB = np.sum(img_tgt*img_fus, axis=0)
    #
    # sam = AB/(A*B)
    # sam = np.arccos(sam)
    # sam = np.mean(sam)*180/3.1415926535
    #
    # return sam
def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[1], -1)
    img_fus = np.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[1], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2))
    B = np.sqrt(np.sum(img_fus**2))
    AB = np.sum(img_tgt*img_fus)

    sam = AB/(A*B)

    sam = np.arccos(sam)
    sam = np.mean(sam)*180/np.pi

    return sam

def calc_ssim(img_tgt, img_fus):
    '''
    :param reference:
    :param target:
    :return:
    '''

    img_tgt = np.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = np.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    ssim = structural_similarity(img_tgt, img_fus, data_range=np.max(img_fus))

    return ssim