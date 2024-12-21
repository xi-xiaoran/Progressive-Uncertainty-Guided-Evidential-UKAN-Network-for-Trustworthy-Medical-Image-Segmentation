import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optin
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from deal_data.dataloader import get_data_loader
from Calibration_train import train_model
from Calibration_test import evaluate_model
from models.Unet import UNet
from monai.networks.nets import BasicUNetPlusPlus, VNet, SwinUNETR, UNETR, AttentionUnet
from models.UKAN import UKAN
from models.PUGEUKAN import PUGEUKAN
from models.SwinUNETR import SwinUNETR
from models.AttentionUnet import AttentionUnet
from models.UNetplusplus import BasicUNetPlusPlus
from models.DUCKNet import DuckNet
from models.EMCAD import EMCADNet
from Loss_functions.Evidence_Loss import EvidentialLoss
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data_name = 'CVC-ClinicDB'
# data_name = 'ETIS'
# data_name = 'Kvasir-SEG'
# data_name = 'ISIC2018'

images_root = 'data/CVC-ClinicDB/PNG/Original'
masks_root = 'data/CVC-ClinicDB/PNG/Ground_Truth'
test_root = 'data/CVC-ClinicDB/Test/test'
GT_root = 'data/CVC-ClinicDB/Test/GT'

# images_root = 'data/ETIS-LaribPolypDB/ETIS-LaribPolypDB/ETIS-LaribPolypDB'
# masks_root = 'data/ETIS-LaribPolypDB/ETIS-LaribPolypDB/Ground_Truth'
# test_root = 'data/ETIS-LaribPolypDB/ETIS-LaribPolypDB//Test'
# GT_root = 'data/ETIS-LaribPolypDB/ETIS-LaribPolypDB//GT'

# images_root = 'data/Kvasir-SEG/images'
# masks_root = 'data/Kvasir-SEG/masks'
# test_root = 'data/Kvasir-SEG/Test'
# GT_root = 'data/Kvasir-SEG/GT'

# images_root = 'data/ISIC/Train'
# masks_root = 'data/ISIC/masks'
# test_root = 'data/ISIC/Test'
# GT_root = 'data/ISIC/GT'
dataset = get_data_loader(images_root, masks_root, channels=3)
random_seed = 14514
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

num_epochs = 100
n_classes = 2
lr = 1e-3
model_name = 'Calibration_UNet'
"""
'Calibration_UKAN'
'Calibration_UNet'
"""
Train = True
Test = False
uncertainty = True
save_path = 'save_models/' + model_name + str(data_name) + '.pth'
seed_everything(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('len dataset:', len(dataset))
print(device)
if Train:
    print('-' * 20, 'train', '-' * 20)
    num = 0
    for train_index, val_index in kf.split(dataset):
        num = num + 1
        print(f'第{num}折交叉验证开始')
        # model = UKAN(input_channels=3,
        #              num_classes=1).to(device)
        model = UNet(
                    spatial_dims=2,
                    in_channels=3,
                    out_channels=1,
                    channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2)
                ).to(device=device)
        # model = SwinUNETR(img_size=(256,256),
        #                   in_channels=3,
        #                   out_channels=1,
        #                   use_checkpoint=True,
        #                   spatial_dims=2).to(device)
        # model = AttentionUnet(spatial_dims=2,
        #                       in_channels=3,
        #                       out_channels=1,
        #                       channels=(32, 64, 128, 256, 512),
        #                       strides=(2, 2, 2, 2)).to(device)
        # model = EMCADNet(num_classes=2).to(device)
        # model = BasicUNetPlusPlus(in_channels=3,out_channels=2,spatial_dims=2).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optin.Adam(model.parameters(), lr=lr)
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        model = train_model(model, train_loader, val_loader, criterion, optimizer, device,
                            num_epochs=num_epochs, zhe=num)

        torch.save(model, save_path)
        # Evaluate the model
        print('-' * 20, 'test', '-' * 20)
        dataset_test = get_data_loader(test_root, GT_root, channels=3)
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
        Dice, Iou, Assd, ECE, UEO = evaluate_model(model, test_loader, device, uncertainty)
        print('Dice', Dice)
        print('Iou', Iou)
        print('ASSD', Assd)
        print('ECE',ECE)
        print('UEO', UEO)
        with open('recording.txt', 'a') as file:
            file.write(f'epoch:{num_epochs}   lr:{lr}   characteristic:{model_name}\n')
            file.write(f'Dice:{Dice}\nIou:{Iou}\nASSD:{Assd}\nECE:{ECE}\nUEO:{UEO}\n')
            file.write(f'-----------------------------------------------\n\n')
        file.close()
        print('txt write completed')
    torch.save(model, save_path)
    print('txt write completed')


else:
    model = torch.load(save_path)

if Test:
    # Evaluate the model
    print('-' * 20, 'test', '-' * 20)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    Dice, Iou, Assd, ECE, UEO = evaluate_model(model, test_loader, device, uncertainty)
    print('Dice', Dice)
    print('Iou', Iou)
    print('ASSD', Assd)
    print('ECE', ECE)
    print('UEO', UEO)
    with open('recording.txt', 'a') as file:
        file.write(f'epoch:{num_epochs}   lr:{lr}   characteristic:{model_name}\n')
        file.write(f'Dice:{Dice}\nIou:{Iou}\nASSD:{Assd}\nECE:{ECE}\nUEO:{UEO}\n')
        file.write(f'-----------------------------------------------\n\n')
    file.close()
    print('txt write completed')








