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
# test_root = 'data/CVC-ClinicDB/PNG/Original'
# GT_root = 'data/CVC-ClinicDB/PNG/Ground_Truth'
# test_root = 'data/Kvasir-SEG/Test'
# GT_root = 'data/Kvasir-SEG/GT'

# test_root = 'data/ETIS-LaribPolypDB/ETIS-LaribPolypDB//Test'
# GT_root = 'data/ETIS-LaribPolypDB/ETIS-LaribPolypDB//GT'

dataset_test = get_data_loader(test_root, GT_root, channels=3)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
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
with torch.no_grad():
    for inputs, labels in test_loader:
        num = num + 1
        inputs, labels = inputs.to(device), labels.to(device)
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
            print(like)
            if (like < likehood):
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                loss = outputs.long() ^ labels.long()
                break
            else:
                u_last = u
        
        outputs = outputs.squeeze(0).squeeze(0).cpu().numpy()
        u = u.squeeze(0).squeeze(0).cpu().numpy()
        loss = loss.squeeze(0).squeeze(0).cpu().numpy()
        plt.subplot(3, 8, 2)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(outputs)
        plt.title('PUGEUKAN', rotation=30)
        plt.subplot(3, 8, 10)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(u)
        plt.subplot(3, 8, 18)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(loss)

        outputs, u, loss = get_out_u_loss(Evid_EMCAD, inputs, labels)
        plt.subplot(3, 8, 3)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(outputs)
        plt.title('EMCAD', rotation=30)
        plt.subplot(3, 8, 11)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(u)
        # plt.title('u')
        plt.subplot(3, 8, 19)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(loss)
        # plt.title('loss')

        outputs, u, loss = get_out_u_loss(Evid_UKAN, inputs, labels)
        plt.subplot(3, 8, 4)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(outputs)
        plt.title('UKAN', rotation=30)
        plt.subplot(3, 8, 12)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(u)
        # plt.title('u')
        plt.subplot(3, 8, 20)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(loss)
        # plt.title('loss')

        outputs, u, loss = get_out_u_loss(Evid_UNet, inputs, labels)
        plt.subplot(3, 8, 5)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(outputs)
        plt.title('UNet', rotation=30)
        plt.subplot(3, 8, 13)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(u)
        # plt.title('u')
        plt.subplot(3, 8, 21)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(loss)
        # plt.title('loss')

        outputs, u, loss = get_out_u_loss(Evid_SwinUNETR, inputs, labels)
        plt.subplot(3, 8, 6)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(outputs)
        plt.title('SwinUNETR', rotation=30)
        plt.subplot(3, 8, 14)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(u)
        # plt.title('u')
        plt.subplot(3, 8, 22)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(loss)
        # plt.title('loss')

        outputs, u, loss = get_out_u_loss(Evid_AttentionUnet, inputs, labels)
        plt.subplot(3, 8, 7)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(outputs)
        plt.title('AttentionUnet', rotation=30)
        plt.subplot(3, 8, 15)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(u)
        # plt.title('u')
        plt.subplot(3, 8, 23)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(loss)
        # plt.title('loss')

        outputs, u, loss = get_out_u_loss(Evid_UNetPlusPlus, inputs, labels)
        plt.subplot(3, 8, 8)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(outputs)
        plt.title('UNet++', rotation=30)
        plt.subplot(3, 8, 16)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(u)
        plt.title('u',x=1.1,y=0.4,rotation=270)
        # plt.title('u')
        plt.subplot(3, 8, 24)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(loss)
        plt.title('loss',x=1.1,y=0,rotation=270)
        # plt.title('loss')

        inputs = inputs.squeeze(0).squeeze(0).permute(1, 2, 0)
        inputs = inputs.cpu().numpy()
        plt.subplot(3, 8, 1)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(inputs)
        plt.title('Input', rotation=30)

        labels = labels.squeeze(0).squeeze(0)
        labels = labels.cpu().numpy()
        plt.subplot(3, 8, 17)
        plt.subplots_adjust(wspace=w, hspace=h)
        plt.axis('off')
        plt.imshow(labels)
        plt.title('GT')
        plt.savefig(f'show/{num}.png')
        plt.close()
        print(f'The {num} th image has been saved')




