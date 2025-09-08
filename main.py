import time
import random
import torch.optim
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from utils import train, val
from Dataset import ChikuseiDataset
from model.SSDT import *
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import scipy.io as scio

torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--image_size', type=int, default=64, help='training patch size')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--n_bands', type=int, default=103, help='output channel number')
    parser.add_argument('--n_feats', type=int, default=48, help='the embed_dim')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--n_epochs', type=int, default=2001, help='end epoch for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=415, help='random seed to use. Default=123')
    parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
    parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
    parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
    parser.add_argument('--local_rank', default=1, type=int, help='None')
    parser.add_argument('--use_distribute', type=int, default=1, help='None')
    parser.add_argument('--dataroot', type=str, default='./data/CAVE_X4/cave')
    parser.add_argument('--dataset', type=str, default='Chikusei', choices=['Chikusei', 'WDCM', 'YRE', 'Pavia'])
    parser.add_argument('--arch', type=str, default='SSDT',
                        choices=['CYformer', 'PSRT', 'ResNet', 'SSRNET', 'DHIF', 'MOGDCN', 'SSFCNN', 'MSDCNN', 'MIMO',
                                 '3DT', 'DSPNet', 'SSDT'])
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='Learning rate weight decay')
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    return args


torch.autograd.set_detect_anomaly(True)
opt = args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(opt)


# 保存模型
def save_checkpoint(model, epoch, data):
    model_out_path = "checkpoints/{}_{}/model_epoch_best.pth.tar".format(opt.arch, data)
    state = {"epoch": epoch, "model": model}
    torch.save(state, model_out_path)

    print("Checkpoints saved to {}".format(model_out_path))


# 加载数据
def dataloader(opt):
    train_set = ChikuseiDataset(opt.dataroot, ratio=opt.upscale_factor, type='train')
    val_set = ChikuseiDataset(opt.dataroot, ratio=opt.upscale_factor, type='eval')
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=4,
                                  pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=4,
                                pin_memory=True)
    return train_loader, val_loader


def main():
    # load data
    print('===> Loading datasets')

    if opt.dataset == 'Chikusei':
        opt.n_bands = 128
        opt.n_bands_rgb = 4
        opt.image_size = 64
        opt.dataroot = ''

    training_data_loader, val_data_loader = dataloader(opt)

    # Build the models
    model = SSDT(opt).cuda()
    Loss = nn.L1Loss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    print('Start Training:')
    t = time.strftime("%Y%m%d%H%M")
    best_psnr = 0
    pbar = tqdm(range(opt.n_epochs))
    for epoch in pbar:
        pbar.set_description_str('Eopch:{},lr:{}'.format(epoch, optimizer.param_groups[0]["lr"]))
        model, loss = train(opt.arch, training_data_loader, model=model, Loss=Loss, optimizer=optimizer,
                                epoch=epoch,
                                scale_factor=opt.upscale_factor)
        print('-' * 20)
        psnr, rmse, ergas, sam = val(opt.arch, val_data_loader, model=model, scale_factor=opt.upscale_factor)
        if (psnr > best_psnr):
            print("Best psnr is: .4f", psnr)
            best_psnr = psnr
            save_checkpoint(model, epoch, t, opt.dataset)
        pbar.set_postfix_str(
            'val PSNR:{:.2f}, val ERGAS:{:.2f}, val RMSE:{:.4f}, val SAM:{:.4f}'.format(psnr,
                                                                                        ergas,
                                                                                        rmse,
                                                                                        sam))

if __name__ == '__main__':
    set_random_seed(415)
    main()
