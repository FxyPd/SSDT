import torch
from torch.autograd import Variable
from torch.nn import functional as F
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
import numpy as np

def train(arch, train_dataloader, model, Loss, optimizer, epoch, scale_factor, criterion=None, loss_=None):
    model.train()
    loss = 0
    for iteration, batch in enumerate(train_dataloader, 1):
        ref, hrmsi, lrhsi = (Variable(batch['hrhsi']).cuda(),
                             Variable(batch['hrmsi']).cuda(),
                             Variable(batch['lrhsi']).cuda())
        input_lr_u = F.interpolate(lrhsi, scale_factor=scale_factor, mode='bicubic')
        out = model(hrmsi, input_lr_u, lrhsi)
        loss = Loss(out, ref)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_dataloader),
                                                                loss.item()))
    return model, loss

def val(arch, val_dataloader, model, scale_factor):
    model.eval()
    with torch.no_grad():
        psnr_list = []
        rmse_list = []
        ergas_list = []
        sam_list = []
        for index, batch in enumerate(val_dataloader):
            ref, hrmsi, lrhsi = (Variable(batch['hrhsi']).cuda(),
                                Variable(batch['hrmsi']).cuda(),
                                Variable(batch['lrhsi']).cuda())
            input_lr_u = F.interpolate(lrhsi, scale_factor=scale_factor, mode='bicubic')
            out = model(hrmsi, input_lr_u, lrhsi)

            ref = ref.detach().cpu().numpy()
            out = out.detach().cpu().numpy()

            psnr = calc_psnr(ref, out)
            rmse = calc_rmse(ref, out)
            ergas = calc_ergas(ref, out)
            sam = calc_sam(ref, out)

            psnr_list.append(psnr)
            rmse_list.append(rmse)
            ergas_list.append(ergas)
            sam_list.append(sam)
    return np.mean(psnr_list), np.mean(rmse), np.mean(ergas), np.mean(sam)

