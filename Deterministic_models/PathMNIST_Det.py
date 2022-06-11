import random, os
import wandb
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.neighbors import KernelDensity
from PIL import Image
import argparse
import importlib
import numpy as np
import pickle
import torch.utils.data
from skimage.util import random_noise
from skimage import io, transform
import logging
import torchattacks
from time import perf_counter
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
from torchvision.utils import save_image
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

import medmnist
from medmnist import INFO, Evaluator


torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(prog="eVI")

# Training Parameters
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs') 
parser.add_argument('--batch_size_train', type=int, default=128, help='Batch size for training')
# Testing Parameters
parser.add_argument('--testing', type=bool, default=True, help='Run inference using the trained model checkpoint')
parser.add_argument('--batch_size_val', type=int, default=1024, help='Batch size for validation')
parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing, can only be 1')
#Adding Noise/Adversarial attack
parser.add_argument('--add_noise', type=bool, default=True, help='Addition of noise during testing')
parser.add_argument('--adv_attack', type=bool, default=False, help='Adding adversarial attack during testing')
parser.add_argument('--adv_trgt', type=int, default=1, help='Adding adversarial attack Target')
parser.add_argument('--var_threshold', type=bool, default=False, help='Addition of noise during testing')
# Load Model
parser.add_argument('--load_model', type=str, default='../../Results/checkpoints/PathMNIST_Det_Res18_model_070322_2100_epoch_100.pth')

parser.add_argument('--data_path', type=str, default='', help='Path to save dataset data. It has train and test subfolders.')

# Learning Rate Parameters
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate') 
parser.add_argument('--wd', type=float, default=0.00001, help='Weight decay') 
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum used for Optimizer')

parser.add_argument('--output_size', type=int, default=9, help='Size of the output')


############################################################
#Deterministic Res18 Model
############################################################
class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        self.stride = stride
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn11 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.stride != 1:
            identity = self.conv11(identity)
            identity = self.bn11(identity) 

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=1, padding=3) #chnaged stride to 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block11 = block(18, 64, 64, stride=1)
        self.block12 = block(18, 64, 64, stride=1)
        self.block21 = block(18, 64, 128, stride=2)
        self.block22 = block(18, 128, 128, stride=1)
        self.block31 = block(18, 128, 256, stride=2)
        self.block32 = block(18, 256, 256, stride=1)
        self.block41 = block(18, 256, 512, stride=2)
        self.block42 = block(18, 512, 512, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.block21(x)
        x = self.block22(x)
        
        x = self.block31(x)
        x = self.block32(x)
        
        x = self.block41(x)
        x = self.block42(x)
    
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
                
        x = self.fc(x)       

        return x    

###########################################################################################################################

def train(args, model, optimizer, criterion, train_loader, epoch):
    print('Training started')
    model.train()
    
    train_losses = []
    train_counter = []
    train_acc = 0
    total_num = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.squeeze().long()
        data, target = data.to(args.devices), target.to(args.devices)
        optimizer.zero_grad()
        mu_o = model(data)        
        loss = criterion(mu_o, target)      
        _, pred = mu_o.max(1, keepdim=True)

        train_acc += pred.eq(target.view_as(pred)).sum().item()
        total_num += len(target)        
        loss.backward()
        optimizer.step()
                
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*args.batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
    acc = train_acc / total_num
    print('Train Accuracy: ', acc)    
    
    if epoch==25 or epoch==50 or epoch==75 or epoch==100:
        torch.save(model.state_dict(), '../../Results/checkpoints/PathMNIST_Det_Res18_model_070322_2100_epoch_{}.pth'.format(epoch))
        torch.save(optimizer.state_dict(), '../../Results/checkpoints/PathMNIST_Det_Res18_optimizer_070322_2100_epoch_{}.pth'.format(epoch))
    
    return loss, acc

#################################################################################

def validation(args, model, valid_loader):
    
    model.eval()
    model.zero_grad()
    val_acc = 0
    total_num = 0
    for idx, (data, targets) in enumerate(valid_loader):
        data, targets = data.to(args.devices), targets.to(args.devices)

        mu_y_out= model.forward(data)

        _, pred = mu_y_out.max(1, keepdim=True)

        val_acc += pred.eq(targets.view_as(pred)).sum().item()
        total_num += len(targets)
      
    acc = val_acc / total_num
    print('Validation Accuracy: ', acc)
    
    return acc

########################################################

def test(args, model, normalize, test_loader):
    
    model.eval()
    model.zero_grad()
    test_acc = 0
    total_num = 0
    for idx, (data, targets) in enumerate(test_loader):
        data, targets = data.to(args.devices), targets.to(args.devices)
        data = normalize(data)

        mu_y_out= model.forward(data)

        _, pred = mu_y_out.max(1, keepdim=True)

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total_num += len(targets)
      
    acc = test_acc / total_num
    print('Clean Test Accuracy: ', acc)
    
    return acc

########################################################

def test_noise(args, model, test_loader, normalize, noise_std):
    print('Starting Test Phase')
    print('Noise_std', noise_std)
    test_acc = 0
    total = 0
    target = list()
    actual, noise_data = list(), list()
    corrmu, imu = list(),list()
    prediction = list()
    predm = list()
    ctr = 0
    
    for idx, (data, targets) in enumerate(test_loader):
        noise = random_noise(data, mode='gaussian', mean=0, var=(noise_std) ** 2, clip=True)
        noisy_img = torch.from_numpy(noise)
        noisy_img, targets = noisy_img.to(args.devices), targets.to(args.devices)
        noisy_img = normalize(noisy_img) 
        
        mu_y_out = model.forward(noisy_img.float())

        _, pred = mu_y_out.max(1, keepdim=True)
        predm.append(pred.detach().cpu().numpy())

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total += len(targets)
        ctr += 1
        mu = nn.functional.softmax(mu_y_out)
              
        correctpred = pred.eq(targets.view_as(pred))
        actual.append(data.numpy())
        noise_data.append(noise)
        target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        prediction.append(mu.detach().cpu().numpy())
        
        if correctpred == True:
            corrmu.append(mu_y_out.detach().cpu().numpy())
        else:
            imu.append(mu_y_out.detach().cpu().numpy())

    acc = 100*test_acc / total
    print('Noisy Test Accuracy: ', acc)
    save_image(noisy_img, 'Figures/Det_Gaussian_img_{}.png'.format(noise_std))

    actual = np.vstack(actual)
    noise_data = np.vstack(noise_data)
    target = np.vstack(target)
    prediction = np.vstack(prediction)
    corrmu = np.vstack(corrmu)
    imu = np.vstack(imu)
    predm = np.vstack(predm)
    
    cf_matrix = confusion_matrix(target, predm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    disp.plot()
    plt.savefig('Figures/Det_Gaussian_Pred_Matrix_{}.png'.format(noise_std))
    
    snr = 10 * np.log10(np.squeeze((np.sum(np.square(actual), (1, 2, 3))) / (np.sum(np.square(actual - noise_data),
                                                                                    (1, 2, 3)))))
    mean_snr = np.mean(snr)    
    print('mean_snr :', mean_snr)
    
    with open('../../Results/withNoise/PathMNIST_Res18/PathMNIST_Res18_Det_Gaussian_Results_070322_2100_{}'.format(noise_std), 'wb') as pf:
        print('saving')

        pickle.dump([prediction, corrmu, imu], pf)
        pf.close()

########################################################
def test_adv(args, new_model, test_loader, attack, eps):
    print('Starting Test Phase for Adv Attack')
    print('Epsilon: ', eps)
    correct = 0
    total = 0
    
    actual = list()
    adv_images, target = list(), list()
    corrmu, imu = list(),list()
    prediction = list()
    corr_adv, corr_target = list(), list()
    i_adv, i_target = list(), list()
    ctr = 0

    for data, targets in test_loader:     
        data, targets = data.to(args.devices), targets.to(args.devices)
        adv_image = attack(data, targets[0])
        mu_y_out = new_model(adv_image)
    
        _, pred = mu_y_out.max(1, keepdim=True)

        correct += pred.eq(targets.view_as(pred)).sum().item()
        total += len(targets)
        ctr += 1
        
        adv_image = adv_image.cpu().numpy()
        correctpred = pred.eq(targets.view_as(pred))
        prediction.append(mu_y_out.detach().cpu().numpy())
        adv_images.append(adv_image)
        target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        actual.append(data.cpu().numpy())
                
        if correctpred == True:
            corrmu.append(mu_y_out.detach().cpu().numpy())            
            corr_adv.append(adv_image)
            corr_target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        else:
            imu.append(mu_y_out.detach().cpu().numpy())            
            i_adv.append(adv_image)
            i_target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        
        if ctr>399:
            print(ctr, 'images') 
            break
      
    acc = 100*correct / total

    actual = np.vstack(actual)
    adv_images = np.vstack(adv_images)
    target = np.vstack(target)
    prediction = np.vstack(prediction)
    corrmu = np.vstack(corrmu)
    imu = np.vstack(imu)
    corr_adv = np.vstack(corr_adv)
    corr_target = np.vstack(corr_target)
    i_adv = np.vstack(i_adv)
    i_target = np.vstack(i_target)
    
    print('Test Accuracy: ', acc)
        
    with open('../../Results/withAdv/PathMNIST_Res18/PathMNIST_Res18_Det_AdvAttk_Results_070322_2100_{}'.format(
        eps), 'wb') as pf:
        print('saving')
        pickle.dump([actual, adv_images, target, prediction, corrmu, imu, corr_adv, i_adv, corr_target, i_target], pf)
        pf.close()
        
########################################################

class Norm(torch.nn.Module):
    def __init__(self, mean, std):
        super(Norm, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1,3,1,1)
        norm_img = (input - mean) / std
        return norm_img
    
############################################################
    
seed_everything(40)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(42)

def main():
    
    wandb.init(project='PathMNIST-Res18-Det')
    args = parser.parse_args()
    wandb.config.update(args)
    args.devices = torch.device('cuda:0')
    print('Using device:', args.devices)      
    
    normalize = transforms.Normalize(mean=[.5], std=[.5]) 
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    data_transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    task = info['task']

    DataClass = getattr(medmnist, info['python_class'])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=False)
    val_dataset = DataClass(split='val', transform=data_transform, download=False)
    test_dataset = DataClass(split='test', transform=data_transform_test, download=False)

    pil_dataset = DataClass(split='train', download=False)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size_train, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size_val, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size_test, shuffle=False)
    
    target_list = torch.tensor(test_dataset.labels)    
    class_count = np.array([len(np.where(target_list == t)[0]) for t in np.unique(target_list)])
    print('class count test: ', class_count)
    
    if args.testing==False:
        network = ResNet(18, Block, 3, args.output_size).to(args.devices)
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)        
        criterion = nn.CrossEntropyLoss() 
        wandb.watch(network)
        t1_start = perf_counter()    
        for epoch in range(1, args.epochs + 1):            
            trg_loss, trg_acc = train(args, network, optimizer, criterion, train_loader, epoch)
            val_acc = validation(args, network, val_loader)
            scheduler.step() 
            wandb.log({'Epoch': epoch, 'Train Accuracy': trg_acc, 'Val Accuracy': val_acc})
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
        test(args, network, normalize, test_loader)
    
    elif args.testing==True:
        print('Initializing Testing')
        network = ResNet(18, Block, 3, args.output_size)
        network.load_state_dict(torch.load(args.load_model))
        print('network loaded')
        logging.info('Model:\n{}'.format(network))
                
        if args.add_noise==True:
            print('Noise testing')
            network = network.to(args.devices)
            network.eval()  
            test(args, network, normalize, test_loader)
            noise_std = [0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15]

            for noise in noise_std:
                test_noise(args, network, test_loader, normalize, noise)

        elif args.adv_attack==True:
            print('Adv Attack testing')
            norm_layer = Norm(mean=[.5, .5, .5], std=[.5, .5, .5])
            new_model = nn.Sequential(norm_layer, network).to(args.devices)
            new_model.eval()
            
            epsilon = [0, 0.001, 0.005, 0.01, 0.1, 1]
            c=0.05        
            for eps in epsilon:
                if eps==0:
                    print('Starting PGD Attack')
                    atk = torchattacks.PGD(new_model, eps=0.005, alpha=2/255, steps=50, random_start=False)
                elif eps==1:
                    print('Starting CW Attack')
                    atk = torchattacks.CW(new_model, c=c, kappa=0, steps=100, lr=0.01)
                else:
                    print('Starting FGSM Attack')
                    atk = torchattacks.FGSM(new_model, eps=eps)
                    
                test_adv(args, new_model, test_loader, atk, eps)
    
if __name__ == '__main__':
    main()    