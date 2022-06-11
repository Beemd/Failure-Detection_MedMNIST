import os
import wandb
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import pandas as pd
from skimage import io, transform
import argparse
import importlib
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import logging
from pytorch_lightning import seed_everything
import torch.multiprocessing
from time import perf_counter
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser(prog="eVI")

# Training Parameters
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size_train', type=int, default=512, help='Batch size for training')
# Testing Parameters
parser.add_argument('--testing', type=bool, default=True, help='Run inference using the trained model checkpoint')
parser.add_argument('--batch_size_val', type=int, default=16384, help='Batch size for validation')
parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing, can only be 1')
# Load Model
parser.add_argument('--load_model', type=str, default='../../Results/checkpoints/OrganaMNIST_var_threshold_model_050422_1700_epoch_50.pth', help='Path to a previously trained model checkpoint')
# Learning Rate Parameters
parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate') 
parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay') 
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum used for Optimizer')
parser.add_argument('--a', type=float, default=1, help='Hyperparameter for loss1')
parser.add_argument('--b', type=float, default=100, help='Hyperparameter for loss2')
parser.add_argument('--c', type=float, default=2, help='Hyperparameter for loss3')

parser.add_argument('--output_size', type=int, default=1, help='Size of the output')

################################################################################################################################
#Threshold Learning Model
##########################################################################

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        
         # Hypers
        self.lr = args.lr
        self.output_size = args.output_size

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(28*28, 1500, bias=False) #OrganaMNIST only has 1 channel
        self.fc2 = nn.Linear(1500, 500, bias=False)
        self.fc3 = nn.Linear(500, 100, bias=False)
        self.fc4 = nn.Linear(100, 5, bias=False)
        self.fc5 = nn.Linear(5, args.output_size, bias=False)
        self.drop = nn.Dropout(0.25)

        
    def forward(self, x):
        mu_flat = torch.flatten(x, start_dim=1)        
        muf = self.fc1(mu_flat)
        muf = self.relu(muf)
        muf = self.drop(muf)
        muf = self.fc2(muf)
        muf = self.relu(muf)
        muf = self.fc3(muf)
        muf = self.relu(muf)
        muf = self.fc4(muf)
        muf = self.relu(muf)
        muf = self.fc5(muf)
        return muf
    
############################################################################################################################### 
#Function for Dataset
############################################################################## 

class Dataset_MedMNIST(Dataset):
    
    def __init__(self, csv_file, imgs_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            imgs_path (string): Path to the directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.imgs = np.load(imgs_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):        
        image = self.imgs[idx]
        labels = self.labels_frame.iloc[idx, 1]
        var = pred = self.labels_frame.iloc[idx, 2]
        pred = self.labels_frame.iloc[idx, 3]
        tag = self.labels_frame.iloc[idx, 4]

        if self.transform:          
            image = self.transform(image)
        
        sample = {'image': image, 'labels': labels, 'var': var,
                  'pred': pred, 'tag': tag}

        return sample

###############################################################################################################################
#Functions for Model training, validation and testing
#############################################################################################

def train(args, model, optimizer, criterion, criterion2, train_loader, epoch):
    model.train()
       
    for sample in train_loader: 
        data = sample['image']
        targets = sample['labels']
        var = sample['var']
        pred = sample['pred']
        tag = sample['tag']
        data, targets = data.to(args.devices), targets.to(args.devices)
        data, targets = data.half(), targets.half()        
        var, pred, tag = var.to(args.devices), pred.to(args.devices), tag.to(args.devices)
        var, pred, tag = var.half(), pred.half(), tag.half()
        
        optimizer.zero_grad()
        predict = model(data)
        predict = torch.reshape(predict, targets.shape)
        
#         loss1 = criterion(predict, targets)
        loss1 = criterion2(predict, targets)
        
        incorrect = torch.where(pred != tag, 1, 0)
        confid = torch.where(predict >= var, 1, 0)        
        incorrconf = incorrect * confid
        threshi = predict*incorrconf
        vari = var*incorrconf
        loss2 = criterion2(threshi, vari)      
        
        correct = torch.where(pred == tag, 1, 0)
        abstain = torch.where(predict < var, 1, 0)  
        corrabs = correct * abstain
        threshc = predict*corrabs
        varc = var*corrabs
        loss3 = criterion2(threshc, varc)

        loss = args.a*loss1 + args.b*loss2 + args.c*loss3        
          
        loss.backward()
        optimizer.step()

    print('epoch {}, Train loss {}'.format(epoch, loss.item()))
    print(loss1)
    print(loss2)
    print(loss3)
    
    if epoch==15 or epoch==25 or epoch==35 or epoch==50:
        torch.save(model.state_dict(), '../../Results/checkpoints/OrganaMNIST_var_threshold_model_050422_1700_epoch_{}.pth'.format(epoch))
        torch.save(optimizer.state_dict(), '../../Results/checkpoints/OrganaMNIST_var_threshold_optimizer_050422_1700_epoch_{}.pth'.format(epoch))
    
    return loss

#################################################################################

def validation(args, model, criterion, criterion2, valid_loader):
    
    model.eval()
    model.zero_grad()

    confidpass = 0
    correctpass = 0
    total = 0
   
    for sample in valid_loader:
        data = sample['image']
        targets = sample['labels']
        var = sample['var']
        pred = sample['pred']
        tag = sample['tag']
        data, targets = data.to(args.devices), targets.to(args.devices)
        data, targets = data.half(), targets.half()
        var, pred, tag = var.to(args.devices), pred.to(args.devices), tag.to(args.devices)
        var, pred, tag = var.half(), pred.half(), tag.half()
        
        predict = model.forward(data)       
        predict = torch.reshape(predict, targets.shape) 
        loss1 = criterion2(predict, targets)
        
        incorrect = torch.where(pred != tag, 1, 0)
        confid = torch.where(predict >= var, 1, 0)        
        incorrconf = incorrect * confid
        threshi = predict*incorrconf
        vari = var*incorrconf
        loss2 = criterion2(threshi, vari)      
        
        correct = torch.where(pred == tag, 1, 0)
        abstain = torch.where(predict < var, 1, 0)  
        corrabs = correct * abstain
        threshc = predict*corrabs
        varc = var*corrabs
        loss3 = criterion2(threshc, varc)

        loss = args.a*loss1 + args.b*loss2 + args.c*loss3    
    
        diff = predict - var
        
        confid = torch.where(predict >= var, 1, 0)
        correct = torch.where(pred == tag, 1, 0)
        corrconfid = confid * correct
        confidpass += torch.sum(confid)
        correctpass += torch.sum(corrconfid)
        total += args.batch_size_val   
    
    print('% Abstained: ', (total - confidpass)/total)    
    print('%Updated accuracy: ', correctpass/confidpass)

    print('Val loss {}'.format(loss.item()))
    print(loss1)
    print(loss2)
    print(loss3)
    

    return loss, correctpass/confidpass

########################################################

def test(args, model, test_loader):
    ctr = 0
    abstain = 0
    corrpass = 0
    incorrabs = 0
    
    for sample in test_loader:
        data = sample['image']
        targets = sample['labels']
        var = sample['var']
        pred = sample['pred']
        tag = sample['tag']
        data, targets = data.to(args.devices), targets.to(args.devices)
        data, targets = data.half(), targets.half()

        var, pred, tag = var.to(args.devices), pred.to(args.devices), tag.to(args.devices)
        var, pred, tag = var.half(), pred.half(), tag.half()

        predict = model.forward(data) 
        predict = torch.reshape(predict, targets.shape)
        
        diff = predict - var
        
        if diff >= 0:
            ctr += 1
            if pred == tag:
                corrpass += 1
        else:
            abstain += 1
            if pred != tag:
                incorrabs += 1
                
    print('Correct passed: ', corrpass)
    print('Incorrect passed: ', ctr - corrpass)
    print('Incorrect abstained: ', incorrabs)
    print('Correct abstained: ', abstain - incorrabs)
    print('% Abstained: ', abstain/(abstain+ctr))
    print('%Updated accuracy: ', corrpass/ctr)    

################################################################################################################################
#Main function
#######################################################################

seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(42)

def main():
    
    wandb.init(project='OrganaMNIST-VarianceThreshold')
    args = parser.parse_args()
    wandb.config.update(args)
    args.devices = torch.device('cuda:0')
    print('Using device:', args.devices)      
    
    normalize = transforms.Normalize(mean=[.5], std=[.5]) 
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize        
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize        
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = Dataset_MedMNIST(
        '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Train_thresh_data.csv',
        '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Train_thresh_imgs_array.npy',
        transform = train_transform)
    valid_dataset = Dataset_MedMNIST(
        '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Val_thresh_data.csv',
        '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Val_thresh_imgs_array.npy',
        transform = valid_transform)
    test_dataset = Dataset_MedMNIST(
        '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Test_NoNoise_thresh_data.csv',
        '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Test_NoNoise_thresh_imgs_array.npy',
#         '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Test_adv_thresh_data_0.csv',
#         '/home/ahmeds57/Projects/SCenE/PytorchCode/Oct_26_2021/VDP/src/MedMNIST/OrganaCSV/Test_adv_thresh_imgs_array_0.npy',
        transform = test_transform)
   
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size_train, num_workers=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size_val, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size_test, num_workers=2, shuffle=False)
    
    if args.testing==False:
        network = MLP(args).to(args.devices)

        network = network.half()

        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)      
        
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        wandb.watch(network)
        t1_start = perf_counter()    
        for epoch in range(1, args.epochs + 1):            
            trg_loss = train(args, network, optimizer, criterion, criterion2, train_loader, epoch)
            val_loss, val_acc = validation(args, network, criterion, criterion2, val_loader)
            scheduler.step()
            wandb.log({'Epoch': epoch, 'Train Loss': trg_loss, 'Val Loss': val_loss, 'Val Acc': val_acc})            
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
        test(args, network, test_loader)
    
    elif args.testing==True:
        print('Initializing Testing')
        network = MLP(args).to(args.devices)
        network = network.half()
        network.load_state_dict(torch.load(args.load_model))
        print('network loaded')
        logging.info('Model:\n{}'.format(network))
        network.eval()

        test(args, network, test_loader)
    
if __name__ == '__main__':
    main()    