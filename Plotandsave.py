import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optin
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.PUGEUKAN import PUGEUKAN

from deal_data.dataloader import get_data_loader
import warnings

warnings.filterwarnings('ignore')

test_root = 'data/CVC-ClinicDB/Test/test'
GT_root = 'data/CVC-ClinicDB/Test/GT'

# test_root = 'data/Kvasir-SEG/Test'
# GT_root = 'data/Kvasir-SEG/GT'

dataset_test = get_data_loader(test_root, GT_root, channels=3)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
# Evid_UAUKAN = torch.load('save_models/Evid_UAUKANCVC-ClinicDB.pth')
Evid_PUGEUKAN = torch.load('save_models/Evid_PUGEUKANCVC-ClinicDB.pth')
Evid_UKAN = torch.load('save_models/Evid_UKANCVC-ClinicDB.pth')
Evid_UNet = torch.load('save_models/Evid_UNetCVC-ClinicDB.pth')
Evid_SwinUNETR = torch.load('save_models/Evid_SwinUNETRCVC-ClinicDB.pth')
Evid_AttentionUnet = torch.load('save_models/Evid_AttentionUnetCVC-ClinicDB.pth')
Evid_UNetPlusPlus = torch.load('save_models/Evid_UNet++CVC-ClinicDB.pth')
Evid_EMCAD = torch.load('save_models/Evid_EMCADCVC-ClinicDB.pth')


def get_out_u_loss(model, inputs, labels):
    outputs = model(inputs)
    alpha = outputs + 1
    S = torch.sum(alpha, dim=1)
    P = alpha / S
    u = 2 / S
    outputs = torch.argmax(P, dim=1)
    loss = outputs.long() ^ labels.long()
    outputs = outputs.squeeze(0).squeeze(0).cpu().numpy()
    u = u.squeeze(0).squeeze(0).detach().cpu().numpy()
    loss = loss.squeeze(0).squeeze(0).cpu().numpy()
    return outputs, u, loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num = 0
turn = 1
w = 0.1
h = 0
Evid_PUGEUKAN.train()
likehood = 0.0001
Evid_PUGEUKAN.clear()
cmap = 'gray'
u_camp = 'jet'
pad_inches = 0.00001
dpi = 300
with torch.no_grad():
    for inputs, labels in test_loader:
        num = num + 1
        if not os.path.exists(f'show/{num}'):
            os.mkdir(f'show/{num}')
        inputs, labels = inputs.to(device), labels.to(device)
        # for i in range(turn):
        #     outputs, u, loss = get_out_u_loss(Evid_UAUKAN, inputs, labels)
        # inputs, labels = inputs.to(device), labels.to(device)
        outputs = Evid_PUGEUKAN(inputs)
        a = outputs + 1
        S = torch.sum(a, dim=1, keepdim=True)
        u = 2 / S
        u_last = u
        while (True):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = Evid_PUGEUKAN(inputs)
            a = outputs + 1
            S = torch.sum(a, dim=1, keepdim=True)
            u = 2 / S
            like = torch.mean(u_last - u).cpu().numpy()
            like = abs(like)
            if (like < likehood):
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                loss = outputs.long() ^ labels.long()
                break
            else:
                u_last = u

        outputs = outputs.squeeze(0).squeeze(0).cpu().numpy()
        u = u.squeeze(0).squeeze(0).cpu().numpy()
        loss = loss.squeeze(0).squeeze(0).cpu().numpy()
        plt.axis('off')
        plt.imshow(outputs,cmap=cmap)
        plt.savefig(f'show/{num}/PUGEUKAN_outputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(u,cmap=u_camp)
        plt.savefig(f'show/{num}/PUGEUKAN_u.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(loss,cmap=cmap)
        plt.savefig(f'show/{num}/PUGEUKAN_loss.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        outputs, u, loss = get_out_u_loss(Evid_EMCAD, inputs, labels)
        plt.axis('off')
        plt.imshow(outputs,cmap=cmap)
        plt.savefig(f'show/{num}/EMCAD_outputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(u,cmap=u_camp)
        plt.savefig(f'show/{num}/EMCAD_u.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(loss,cmap=cmap)
        plt.savefig(f'show/{num}/EMCAD_loss.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        outputs, u, loss = get_out_u_loss(Evid_UKAN, inputs, labels)
        plt.axis('off')
        plt.imshow(outputs,cmap=cmap)
        plt.savefig(f'show/{num}/UKAN_outputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(u,cmap=u_camp)
        plt.savefig(f'show/{num}/UKAN_u.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(loss,cmap=cmap)
        plt.savefig(f'show/{num}/UKAN_loss.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        outputs, u, loss = get_out_u_loss(Evid_UNet, inputs, labels)
        plt.axis('off')
        plt.imshow(outputs,cmap=cmap)
        plt.savefig(f'show/{num}/UNet_outputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(u,cmap=u_camp)
        plt.savefig(f'show/{num}/UNet_u.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(loss,cmap=cmap)
        plt.savefig(f'show/{num}/UNet_loss.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        outputs, u, loss = get_out_u_loss(Evid_SwinUNETR, inputs, labels)
        plt.axis('off')
        plt.imshow(outputs,cmap=cmap)
        plt.savefig(f'show/{num}/SwinUNETR_outputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(u,cmap=u_camp)
        plt.savefig(f'show/{num}/SwinUNETR_u.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(loss,cmap=cmap)
        plt.savefig(f'show/{num}/SwinUNETR_loss.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        outputs, u, loss = get_out_u_loss(Evid_AttentionUnet, inputs, labels)
        plt.axis('off')
        plt.imshow(outputs,cmap=cmap)
        plt.savefig(f'show/{num}/AttentionUnet_outputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(u,cmap=u_camp)
        plt.savefig(f'show/{num}/AttentionUnet_u.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(loss,cmap=cmap)
        plt.savefig(f'show/{num}/AttentionUnet_loss.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        outputs, u, loss = get_out_u_loss(Evid_UNetPlusPlus, inputs, labels)
        plt.axis('off')
        plt.imshow(outputs,cmap=cmap)
        plt.savefig(f'show/{num}/UNetPlusPlus_outputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(u,cmap=u_camp)
        plt.savefig(f'show/{num}/UNetPlusPlus_u.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        plt.axis('off')
        plt.imshow(loss,cmap=cmap)
        plt.savefig(f'show/{num}/UNetPlusPlus_loss.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        inputs = inputs.squeeze(0).squeeze(0).permute(1, 2, 0)
        inputs = inputs.cpu().numpy()
        plt.axis('off')
        plt.imshow(inputs,cmap=cmap)
        plt.savefig(f'show/{num}/Inputs.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()

        labels = labels.squeeze(0).squeeze(0)
        labels = labels.cpu().numpy()
        plt.axis('off')
        plt.imshow(labels,cmap=cmap)
        plt.savefig(f'show/{num}/labels.png', pad_inches=pad_inches, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f'The {num} th image has been saved')




