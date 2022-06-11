import random, os
import wandb
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import argparse
import importlib
import numpy as np
import pickle
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
from skimage.util import random_noise
from skimage import io, transform
import logging
import torchattacks
from time import perf_counter
import pandas as pd
import torch.utils.data as data
from torchvision.utils import save_image
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

import medmnist
from medmnist import INFO, Evaluator

from VDPLayers_CoVar_DermaMNIST import VDP_Flatten, VDP_Conv2D, VDP_Relu, VDP_Maxpool, VDP_FullyConnected, VDP_Softmax, VDP_BatchNorm2D, VDP_AdaptiveAvgPool2d

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(prog="eVI")

# Training Parameters
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs') 
parser.add_argument('--batch_size_train', type=int, default=32, help='Batch size for training')
# Testing Parameters
parser.add_argument('--testing', type=bool, default=True, help='Run inference using the trained model checkpoint')
parser.add_argument('--batch_size_val', type=int, default=30, help='Batch size for validation')
parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing, can only be 1')
#Adding Noise/Adversarial attack
parser.add_argument('--add_noise', type=bool, default=False, help='Addition of noise during testing')
parser.add_argument('--adv_attack', type=bool, default=True, help='Adding adversarial attack during testing')
parser.add_argument('--adv_trgt', type=int, default=0, help='Adding adversarial attack Target')
parser.add_argument('--var_threshold', type=bool, default=False, help='Manual Thresholding')
#Data Generation for Variance Threshold Learning
parser.add_argument('--datagen', type=bool, default=False, help='Learned Thresholding')
parser.add_argument('--adv_datagen', type=bool, default=False, help='Testing Thresholding on Adversarial images')
# Load Model
parser.add_argument('--load_model', type=str, default='../../Results/checkpoints/OrganaMNIST_VDP_Res18_model_040522_1200_epoch_100.pth')

# Loss Function Parameters
parser.add_argument('--tau_conv1', type=float, default=0.005842016984390643, help='KL Weight Term') 
parser.add_argument('--tau_b2', type=float, default=0.0029019904917542173, help='KL Weight Term') 
parser.add_argument('--tau_b3', type=float, default=0.0008835503985268273, help='KL Weight Term') 
parser.add_argument('--tau_b4', type=float, default=0.003504197145529926, help='KL Weight Term') 
parser.add_argument('--tau_fc', type=float, default=0.006731325522510915, help='KL Weight Term') 
parser.add_argument('--clamp', type=float, default=1000, help='Clamping')
# Learning Rate Parameters
parser.add_argument('--lr', type=float, default=0.00048106976101568605, help='Learning Rate') 
parser.add_argument('--wd', type=float, default=6.652706764971839e-06, help='Weight decay') 
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum used for Optimizer')

parser.add_argument('--output_size', type=int, default=11, help='Size of the output')


############################################################
#VDP Model
############################################################
class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, stride=1, input_flag=False):
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
        self.conv11 = VDP_Conv2D(in_channels, out_channels, kernel_size=1, stride=2)
        self.bn11 = VDP_BatchNorm2D(out_channels)
        self.relu_ = VDP_Relu()

    def forward(self, x, sig):
        identity = x
        sigma = sig
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
            identity, sigma = self.conv11(identity, sigma)
            identity, sigma = self.bn11(identity, sigma)      

        x += identity
        x, sigma = self.relu_(x, sigma)
        return x, sigma


class ResNet(nn.Module):
    def __init__(self, args, num_layers, block, image_channels, num_classes):
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
        self.tau_conv1 = args.tau_conv1
        self.tau_b2 = args.tau_b2
        self.tau_b3 = args.tau_b3
        self.tau_b4 = args.tau_b4
        self.tau_fc = args.tau_fc
        
        self.output_size = num_classes
        self.conv1 = VDP_Conv2D(image_channels, 64, kernel_size=7, stride=1, padding=3, input_flag=True)
        self.bn1 = VDP_BatchNorm2D(64)
        self.relu = VDP_Relu()
        self.maxpool = VDP_Maxpool(kernel_size=3, stride=1, padding=1)
        self.block11 = block(18, 64, 64, stride=1)
        self.block12 = block(18, 64, 64, stride=1)
        self.block21 = block(18, 64, 128, stride=2)
        self.block22 = block(18, 128, 128, stride=1)
        self.block31 = block(18, 128, 256, stride=2)
        self.block32 = block(18, 256, 256, stride=1)
        self.block41 = block(18, 256, 512, stride=2)
        self.block42 = block(18, 512, 512, stride=1)

        self.avgpool = VDP_AdaptiveAvgPool2d()
        self.fc = VDP_FullyConnected(512 * self.expansion, num_classes)
        self.flatten = VDP_Flatten()
        self.softmax = VDP_Softmax()

    def forward(self, x):
        x, sigma = self.conv1(x)
                
        x, sigma = self.bn1(x, sigma)
        x, sigma = self.relu(x, sigma)
        x, sigma = self.maxpool(x, sigma)
        
        x, sigma = self.block11(x, sigma)
        x, sigma = self.block12(x, sigma)
        
        x, sigma = self.block21(x, sigma)
        x, sigma = self.block22(x, sigma)
        
        x, sigma = self.block31(x, sigma)
        x, sigma = self.block32(x, sigma)
        
        x, sigma = self.block41(x, sigma)
        x, sigma = self.block42(x, sigma)
    
        x, sigma = self.avgpool(x, sigma)
        x, sigma = self.flatten(x, sigma)
        
        x, sigma = self.fc(x, sigma)
        x, sigma = self.softmax(x, sigma)

        return x, sigma
    
    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        NS = torch.diag(torch.ones(self.output_size, device=y_pred_sd.device) * torch.tensor(
            0.001, device=y_pred_sd.device))
        y_pred_sd_inv = torch.inverse(y_pred_sd + NS)
        mu_ = y_pred_mean - y_test
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        mu_sigmainv_mu = (torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1)).mean()
        logdet = ((torch.slogdet(y_pred_sd + NS)[1]).unsqueeze(1)).mean()
        ms = (torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1) +
              (torch.slogdet(y_pred_sd + NS)[1]).unsqueeze(1)).mean()
        return ms, mu_sigmainv_mu, logdet
    
    def batch_loss(self, output_mean, output_sigma, target):
        output_sigma_clamp = torch.clamp(output_sigma,-1000,1000)
        neg_log_likelihood, mu_2_siginv, logdet = self.nll_gaussian(output_mean, output_sigma_clamp, target)
        loss_value = 0.08*neg_log_likelihood + (self.tau_conv1*self.conv1.kl_loss_term() +
                                                self.tau_b2*self.block21.conv11.kl_loss_term() + 
                                                self.tau_b3*self.block31.conv11.kl_loss_term() +
                                                self.tau_b4*self.block41.conv11.kl_loss_term() +
                                                self.tau_fc*self.fc.kl_loss_term())
       
        return loss_value, self.conv1.kl_loss_term(), self.block21.conv11.kl_loss_term(), self.block31.conv11.kl_loss_term(), self.block41.conv11.kl_loss_term(), self.fc.kl_loss_term(), neg_log_likelihood, mu_2_siginv, logdet

###########################################################################################################################

class SelectOutput(nn.Module):
    def __init__(self):
        super(SelectOutput, self).__init__()

    def forward(self,x):
        out = x[0]
        return out

########################################################    
   
def train(args, model, optimizer, train_loader, epoch):
    model.train()
    
    train_losses = []
    train_counter = []
    nll = list()
    kl1, kl2, kl3, kl4, kl5 = list(), list(), list(), list(), list()
    mu2_sigI, LDet = list(), list()
    train_acc = 0
    total_num = 0
    print('Training...')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.devices), target.to(args.devices)
        optimizer.zero_grad()
        mu_o, sigma_o = model(data)
        labels = (nn.functional.one_hot(target, args.output_size)).squeeze()        
        loss, klo1, klo2, klo3, klo4, klo5, nlloss, mus_sinv, ldet = model.batch_loss(mu_o, sigma_o, labels) 
        kl1.append(klo1.detach().cpu().numpy())
        kl2.append(klo2.detach().cpu().numpy())
        kl3.append(klo3.detach().cpu().numpy())
        kl4.append(klo4.detach().cpu().numpy())
        kl5.append(klo5.detach().cpu().numpy())
        nll.append(nlloss.detach().cpu().numpy())
        mu2_sigI.append(mus_sinv.detach().cpu().numpy()) 
        LDet.append(ldet.detach().cpu().numpy())
        
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
    kl1 = np.vstack(kl1)
    kl1 = np.mean(kl1)
    kl2 = np.vstack(kl2)
    kl2 = np.mean(kl2)
    kl3 = np.vstack(kl3)
    kl3 = np.mean(kl3)
    kl4 = np.vstack(kl4)
    kl4 = np.mean(kl4)
    kl5 = np.vstack(kl5)
    kl5 = np.mean(kl5)
    nll = np.vstack(nll)
    nll = np.mean(nll)
    mu2_sigI = np.vstack(mu2_sigI)
    mu2_sigI = np.mean(mu2_sigI)
    LDet = np.vstack(LDet)
    LDet = np.mean(LDet)
    
    if epoch==25 or epoch==50 or epoch==75 or epoch==100 or epoch==125 or epoch==150 or epoch==175 or epoch==200:
        torch.save(model.state_dict(), '../../Results/checkpoints/OrganaMNIST_VDP_Res18_model_040522_1200_epoch_{}.pth'.format(epoch))
        torch.save(optimizer.state_dict(), '../../Results/checkpoints/OrganaMNIST_VDP_Res18_optimizer_040522_1200_epoch_{}.pth'.format(epoch))
    
    return loss, acc, kl1, kl2, kl3, kl4, kl5, nll, mu2_sigI, LDet

#################################################################################

def validation(args, model, valid_loader):    
    model.eval()
    model.zero_grad()
    val_acc = 0
    total_num = 0
    for idx, (data, targets) in enumerate(valid_loader):
        data, targets = data.to(args.devices), targets.to(args.devices)

        mu_y_out, sigma_y_out = model.forward(data)

        _, pred = mu_y_out.max(1, keepdim=True)

        val_acc += pred.eq(targets.view_as(pred)).sum().item()
        total_num += len(targets)
      
    acc = val_acc / total_num
    print('Validation Accuracy: ', acc)
    
    return acc
#################################################################################

def test(args, model, normalize, valid_loader):    
    model.eval()
    model.zero_grad()
    test_acc = 0
    total_num = 0
    for idx, (data, targets) in enumerate(valid_loader):
        data, targets = data.to(args.devices), targets.to(args.devices)
        data = normalize(data)

        mu_y_out, sigma_y_out = model.forward(data)

        _, pred = mu_y_out.max(1, keepdim=True)

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total_num += len(targets)
      
    acc = test_acc / total_num
    print('Clean Test Accuracy: ', acc)
    
    return acc

########################################################

def test_noise(args, model, test_loader, normalize, noise_std):
    print('Starting Test Phase')
    test_acc = 0
    total = 0
    target = list()
    actual, noise_data = list(), list()
    corrmu, imu = list(),list()
    corrvar, ivar = list(), list()
    prediction, variance = list(), list()
    predm = list()
    ctr = 0
    
    for idx, (data, targets) in enumerate(test_loader):
        
        noise = random_noise(data, mode='gaussian', mean=0, var=(noise_std) ** 2, clip=True)
        noisy_img = torch.from_numpy(noise)
        noisy_img, targets = noisy_img.to(args.devices), targets.to(args.devices)
        noisy_img = normalize(noisy_img)

        mu_y_out, sigma_y_out = model.forward(noisy_img.float())

        _, pred = mu_y_out.max(1, keepdim=True)
        predm.append(pred.detach().cpu().numpy())

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total += len(targets)
        ctr += 1
        
        correctpred = pred.eq(targets.view_as(pred))
        actual.append(data.numpy())
        noise_data.append(noise)
        target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        prediction.append(mu_y_out.detach().cpu().numpy())
        variance.append(sigma_y_out.detach().cpu().numpy())
        
        if correctpred == True:
            corrmu.append(mu_y_out.detach().cpu().numpy())
            corrvar.append(sigma_y_out.detach().cpu().numpy())
        else:
            imu.append(mu_y_out.detach().cpu().numpy())
            ivar.append(sigma_y_out.detach().cpu().numpy())

    acc = 100*test_acc / total
    print('Noise_std: ', noise_std)
    print('Test Accuracy Noisy: ', acc)

    actual = np.vstack(actual)
    noise_data = np.vstack(noise_data)
    target = np.vstack(target)
    prediction = np.vstack(prediction)
    corrmu = np.vstack(corrmu)
    print('number of  correct pred:', len(corrmu))
    corrvar = np.vstack(corrvar)
    imu = np.vstack(imu)
    print('number of incorrect pred:', len(imu))
    ivar = np.vstack(ivar)
    predm = np.vstack(predm)
    variance = np.vstack(variance)
    
    cf_matrix = confusion_matrix(target, predm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    disp.plot()
    plt.savefig('OrganaFigures/VDP/VDP_GN_Pred_Matrix_{}.png'.format(noise_std))
    
    snr = 10 * np.log10(np.squeeze((np.sum(np.square(actual), (1, 2, 3))) / (np.sum(np.square(actual - noise_data),
                                                                                    (1, 2, 3)))))
    mean_snr = np.mean(snr)    
    print('mean_snr :', mean_snr)
    
    with open('../../Results/withNoise/OrganaMNIST_Res18/OrganaMNIST_Res18_VDP_GN_Results_040522_1200_{}'.format(noise_std), 'wb') as pf:
        print('saving')
        pickle.dump([actual, noise_data, target, prediction, variance, corrmu, imu, corrvar, ivar], pf)
        pf.close()
        
########################################################

def test_no_noise(args, model, test_loader, normalize):
    print('Starting Test Phase for No Noise')
    test_acc = 0
    total = 0
    target = list()
    actual, noise_data = list(), list()
    corrmu, imu = list(),list()
    corrvar, ivar = list(), list()
    prediction, variance = list(), list()
    predm = list()
    noise_std = 0
    
    for idx, (data, targets) in enumerate(test_loader):
        data, targets = data.to(args.devices), targets.to(args.devices)
        data = normalize(data)

        mu_y_out, sigma_y_out = model.forward(data)

        _, pred = mu_y_out.max(1, keepdim=True)
        predm.append(pred.detach().cpu().numpy())

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total += len(targets)
        
        correctpred = pred.eq(targets.view_as(pred))
        actual.append(data.cpu().numpy())
        noise_data.append(data.cpu().numpy())
        target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        prediction.append(mu_y_out.detach().cpu().numpy())
        variance.append(sigma_y_out.detach().cpu().numpy())
        
        if correctpred == True:
            corrmu.append(mu_y_out.detach().cpu().numpy())
            corrvar.append(sigma_y_out.detach().cpu().numpy())
        else:
            imu.append(mu_y_out.detach().cpu().numpy())
            ivar.append(sigma_y_out.detach().cpu().numpy())

    acc = 100*test_acc / total
    print('Noise_std: ', noise_std)
    print('Test Accuracy No Noise: ', acc)

    actual = np.vstack(actual)
    noise_data = np.vstack(noise_data)
    target = np.vstack(target)
    prediction = np.vstack(prediction)
    corrmu = np.vstack(corrmu)
    corrvar = np.vstack(corrvar)
    imu = np.vstack(imu)
    ivar = np.vstack(ivar)
    predm = np.vstack(predm)
    variance = np.vstack(variance)
    
    cf_matrix = confusion_matrix(target, predm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    disp.plot()
    plt.savefig('OrganaFigures/VDP/VDP_GN_Pred_Matrix_{}.png'.format(noise_std))
    
    snr = 10 * np.log10(np.squeeze((np.sum(np.square(actual), (1, 2, 3))) / (np.sum(np.square(actual - noise_data),
                                                                                    (1, 2, 3)))))
    mean_snr = np.mean(snr)    
    print('mean_snr :', mean_snr)
    
    with open('../../Results/withNoise/OrganaMNIST_Res18/OrganaMNIST_Res18_VDP_GN_Results_040522_1200_{}'.format(noise_std), 'wb') as pf:
        print('saving')
        pickle.dump([actual, noise_data, target, prediction, variance, corrmu, imu, corrvar, ivar], pf)
        pf.close()
        
########################################################

def test_adv(args, new_model, test_loader, attack, eps):
    print('Starting Test Phase for Adv Attack')
    print('Epsilon: ', eps)
#     print('Target: ', adv_trgt)
    test_acc = 0
    total = 0
    
    actual = list()
    adv_images, target = list(), list()
    corrmu, imu = list(),list()
    corrvar, ivar = list(), list()
    prediction, variance = list(), list()
    corr_adv, corr_target = list(), list()
    i_adv, i_target = list(), list()
    ctr = 0
    
    for data, targets in test_loader:    
        data, targets = data.to(args.devices), targets.to(args.devices)
        adv_image = attack(data, targets[0])
        
        mu_y_out, sigma_y_out = new_model(adv_image)
    
        _, pred = mu_y_out.max(1, keepdim=True)

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total += len(targets)
        ctr += 1
        
        adv_image = adv_image.cpu().numpy()
        correctpred = pred.eq(targets.view_as(pred))
        prediction.append(mu_y_out.detach().cpu().numpy())
        adv_images.append(adv_image)
        target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        actual.append(data.cpu().numpy())
        variance.append(sigma_y_out.detach().cpu().numpy())
        
        if correctpred == True:
            corrmu.append(mu_y_out.detach().cpu().numpy())
            corrvar.append(sigma_y_out.detach().cpu().numpy())
            corr_adv.append(adv_image)
            corr_target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        else:
            imu.append(mu_y_out.detach().cpu().numpy())
            ivar.append(sigma_y_out.detach().cpu().numpy())
            i_adv.append(adv_image)
            i_target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        
        if ctr>399:
            print(ctr, 'images') 
            break
      
    acc = 100*test_acc / total


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
    variance = np.vstack(variance)
    corrvar = np.stack(corrvar)
    ivar = np.stack(ivar)
    
    print('Test Accuracy: ', acc)

    with open('../../Results/withAdv/OrganaMNIST_Res18/OrganaMNIST_Res18_VDP_AdvAttk_Results_040522_1200_{}_{}'.format(
        eps, args.adv_trgt), 'wb') as pf:
        print('saving')
        pickle.dump([actual, adv_images, target, prediction, corrmu, imu, corr_adv, i_adv, corr_target,
                     i_target, variance, corrvar, ivar], pf)
        pf.close()

########################################################

def data_gen_eval(args, model, normalize, train_loader):
    print('Dataset for Threshold Learning Evaluation')
    imgs, labels = list(), list()
    predm, sigm, sigclean = list(), list(), list()
    variance = list()
    res_lst = list()
    img_lst = list()
    col = ['var_clean', 'variance', 'prediction', 'labels']
    res_lst.append(col)
    noises = [0.3]
#     noises = [0.001, 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

    for idx, (data, targets) in enumerate(train_loader):        
                       
        data_norm = normalize(data)
        data_norm, targets = data_norm.to(args.devices), targets.to(args.devices)
        mu, sigma = model.forward(data_norm)
        _, pred = mu.max(1, keepdim=True)
        
        fix_indices = pred * sigma.shape[-1]    
        sigma_indices = pred + fix_indices
        sig_max = torch.gather(sigma.reshape(pred.shape[0], -1), 1, sigma_indices)
        
#         imgs.append(data.numpy())
#         labels.append(targets[0].cpu().numpy())
#         predm.append(pred.detach().cpu().numpy())
#         sigm.append(sig_max.detach().cpu().numpy())
#         sigclean.append(sig_max.detach().cpu().numpy()) 

        for noise_std in noises:
            noise = random_noise(data, mode='gaussian', mean=0, var=(noise_std) ** 2, clip=True)            
            noisy_img = torch.from_numpy(noise)         
            noisy_img = noisy_img.to(args.devices)
            noisy_img = normalize(noisy_img) 
            
            mu_y_out, sigma_y_out = model.forward(noisy_img.float())      
            _, pred_ = mu_y_out.max(1, keepdim=True)  
                                        
            fix_indices_ = pred_ * sigma_y_out.shape[-1]    
            sigma_indices_ = pred_ + fix_indices_
            sig_max_ = torch.gather(sigma_y_out.reshape(pred_.shape[0], -1), 1, sigma_indices_)                        

            imgs.append(noise)
            labels.append(targets[0].cpu().numpy())
            predm.append(pred_.detach().cpu().numpy())
            sigm.append(sig_max_.detach().cpu().numpy())
            sigclean.append(sig_max.detach().cpu().numpy())
    
    imgs = np.vstack(imgs)
    labels = np.vstack(labels)
    predm = np.vstack(predm)
    sigm = np.vstack(sigm)
    sigclean = np.vstack(sigclean)
    result = np.hstack((sigclean, sigm, predm, labels))
    res_lst.append(result)
    res_array = np.vstack(res_lst)
    df = pd.DataFrame(res_array)
    df.to_csv('OrganaCSV/Test_HighNoise_thresh_data.csv', header=False) 
    np.save('OrganaCSV/Test_HighNoise_thresh_imgs_array', imgs)
    print('Completed')

########################################################

def data_gen_adv(args, model, normalize, test_loader, attack, eps):
    print('Dataset for Adversarial Threshold Testing')
    imgs, labels = list(), list()
    predm, sigm, sigclean = list(), list(), list()
    res_lst = list()
    img_lst = list()
    col = ['var_clean', 'variance', 'prediction', 'labels']
    res_lst.append(col)
    ctr=0
    
    for idx, (data, targets) in enumerate(test_loader):        
                       
        data, targets = data.to(args.devices), targets.to(args.devices)
        adv_image = attack(data, targets[0])
        
        mu, sigma = model(adv_image)
        _, pred = mu.max(1, keepdim=True)
        correctpred = pred.eq(targets.view_as(pred))
        
        fix_indices = pred * sigma.shape[-1]    
        sigma_indices = pred + fix_indices
        sig_max = torch.gather(sigma.reshape(pred.shape[0], -1), 1, sigma_indices)      
         
        imgs.append(adv_image.cpu().numpy())
        labels.append(targets[0].cpu().numpy())
        predm.append(pred.detach().cpu().numpy())
        sigm.append(sig_max.detach().cpu().numpy())
        sigclean.append(sig_max.detach().cpu().numpy())
        ctr += 1
        if ctr>399:
            print(ctr, 'images') 
            break
    
    imgs = np.vstack(imgs)
    labels = np.vstack(labels)
    predm = np.vstack(predm)
    sigm = np.vstack(sigm)
    sigclean = np.vstack(sigclean)

    result = np.hstack((sigclean, sigm, predm, labels))
    res_lst.append(result)
    res_array = np.vstack(res_lst)
    df = pd.DataFrame(res_array)
    df.to_csv('OrganaCSV/Test_adv_thresh_data_{}.csv'.format(eps), header=False) 
    np.save('OrganaCSV/Test_adv_thresh_imgs_array_{}'.format(eps), imgs)
    print('Completed')

########################################################

class Norm(torch.nn.Module):
    def __init__(self, mean, std):
        super(Norm, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 1, 1, 1)
        std = self.std.reshape(1,1,1,1)
        norm_img = (input - mean) / std
        return norm_img

########################################################
                
seed_everything(45)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(45)

def main():
    
    wandb.init(project='OrganaMNIST-Res18-VDP')
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
    
    data_flag = 'organamnist'
    info = INFO[data_flag]
    task = info['task']

    DataClass = getattr(medmnist, info['python_class'])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=False)
    val_dataset = DataClass(split='val', transform=data_transform, download=False)
    test_dataset = DataClass(split='test', transform=data_transform_test, download=False)
    train_dataset_2 = DataClass(split='train', transform=data_transform_test, download=False)
    val_dataset_2 = DataClass(split='val', transform=data_transform_test, download=False)

    pil_dataset = DataClass(split='train', download=False)
    
    # create a weighted sampler
    target_list = torch.tensor(train_dataset.labels)
    target_list = target_list.squeeze()
    target_list = target_list.long()
    class_count = np.array([len(np.where(target_list == t)[0]) for t in np.unique(target_list)])
    print('Train class count: ', class_count)
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all, num_samples=len(class_weights_all))

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size_train,
                                   sampler=weighted_sampler)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=args.batch_size_val, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size_test, shuffle=False)
    datagen_loader = data.DataLoader(dataset=train_dataset_2, batch_size=args.batch_size_test, shuffle=False)
    valdata_loader = data.DataLoader(dataset=val_dataset_2, batch_size=args.batch_size_test, shuffle=False)   
   
    if args.testing==False:
        network = ResNet(args, 18, Block, 1, args.output_size).to(args.devices)
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1) #Previous training on 30, 60 with 50 epochs
        wandb.watch(network)        
        t1_start = perf_counter() 
        print('Training started')
        
        for epoch in range(1, args.epochs + 1):
            trg_loss, trg_acc, kl1, kl2, kl3, kl4, kl5, nll, mu2_sinv, ldet = train(args, network, optimizer,
                                                                                    train_loader, epoch)
            val_acc = validation(args, network, val_loader)
            scheduler.step()
            wandb.log({'Epoch': epoch, 'Train Accuracy': trg_acc, 'Val Accuracy': val_acc, 'Training Loss': trg_loss,
                       'KLConv1': kl1, 'KLConvB2': kl2, 'KLConvB3': kl3, 'KLConvB4': kl4, 'KLFC': kl5, 'NLL': nll,
                       'Loss_SigInv': mu2_sinv, 'LogDetSig': ldet
                       })
        test(args, network, normalize, test_loader)
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
    
    elif args.testing==True:
        print('Initializing Testing')
        network = ResNet(args, 18, Block, 1, args.output_size)
        network.load_state_dict(torch.load(args.load_model))
        print('network loaded')
        logging.info('Model:\n{}'.format(network))
        
        if args.var_threshold==True:
            print('Threshold Variance Testing')
            network.eval()
            network = network.to(args.devices)
            
            test_varthresh(args, network, normalize, test_loader)
        
        elif args.datagen==True:
            network = network.to(args.devices)
            network.eval()
            #Use the appropriate data loader for generating train(datagen_loader), val(valdata_loader) or test(test_loader)   dataset for threshold learning.
#             data_gen(args, network, normalize, datagen_loader)
            data_gen_eval(args, network, normalize, test_loader)
        
        elif args.adv_datagen==True:            
            norm_layer = Norm(mean=[.5], std=[.5])
            new_model = nn.Sequential(norm_layer, network, SelectOutput()).to(args.devices)
            new_model_2 = nn.Sequential(norm_layer, network).to(args.devices)
            new_model.eval()
            new_model_2.eval()
            
            epsilon = [0, 0.005, 0.02, 0.04, 1.20, 1]
            
            c=0.95      
            for eps in epsilon:
                if eps==0:
                    print('Starting PGD Attack')
                    atk = torchattacks.PGD(new_model, eps=0.02, alpha=2/255, steps=50, random_start=False)
                elif eps==1:
                    print('Starting CW Attack')
                    atk = torchattacks.CW(new_model, c=c, kappa=0, steps=100, lr=0.01)
                else:
                    print('Starting FGSM Attack')
                    atk = torchattacks.FGSM(new_model, eps=eps)
            
                data_gen_adv(args, new_model_2, normalize, test_loader, atk, eps)        
        
        elif args.add_noise==True:
            print('Noise testing')
            network = network.to(args.devices)
            network.eval()
            
            noise_std = [0, 0.001, 0.01, 0.05, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.6]
            for noise in noise_std:
                if noise == 0:
                    test_no_noise(args, network, test_loader, normalize)
                else:
                    test_noise(args, network, test_loader, normalize, noise)

        elif args.adv_attack==True:
            print('Adv Attack testing')
            norm_layer = Norm(mean=[.5], std=[.5])
            new_model = nn.Sequential(norm_layer, network, SelectOutput()).to(args.devices)
            new_model_2 = nn.Sequential(norm_layer, network).to(args.devices)
            new_model.eval()
            new_model_2.eval()
            
            epsilon = [0, 0.005, 0.02, 0.04, 1.20, 1]
            c=0.95         
            for eps in epsilon:
                if eps==0:
                    print('Starting PGD Attack')
                    atk = torchattacks.PGD(new_model, eps=0.02, alpha=2/255, steps=50, random_start=False)
                elif eps==1:
                    print('Starting CW Attack')
                    atk = torchattacks.CW(new_model, c=c, kappa=0, steps=100, lr=0.01)
                else:
                    print('Starting FGSM Attack')
                    atk = torchattacks.FGSM(new_model, eps=eps)
                
                test_adv(args, new_model_2, test_loader, atk, eps)
    
if __name__ == '__main__':
    main()    