import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mc
from scipy import stats
from PIL import Image
import argparse

def main_run():
    
    parser = argparse.ArgumentParser(prog="eVI")

# Folder/Files
#     parser.add_argument('--folder', type=str, default='PathMNIST_070322_2300', help='Number of epochs')
#     parser.add_argument('--filename', type=str, default='PathMNIST_070322_2300', help='Batch size for training')
    args = parser.parse_args()

    eps = 2
    trgt = 0
#     pf = open('../../Results/withAdv/PathMNIST_Res18/PathMNIST_Res18_VDP_AdvAttk_Results_070322_2300_{}_{}'.format(
#         eps, trgt), 'rb')
#     pf = open('../../Results/withAdv/DermaMNIST_Res18/DermaMNIST_Res18_VDP_AdvAttk_Results_090322_1600_{}_{}'.format(
#         eps, trgt), 'rb')
    pf = open('../../Results/withAdv/OrganaMNIST_Res18/OrganaMNIST_Res18_VDP_AdvAttk_Results_250322_1200_{}_{}'.format(
        eps, trgt), 'rb')
    
    actual, adv_images, target, prediction, corrmu, imu, corr_adv, i_adv, corr_target, i_target, variance, corrvar, ivar = pickle.load(pf)
    pf.close()
    
    snr = 10 * np.log10(np.squeeze((np.sum(np.square(actual), (1, 2, 3))) / (np.sum(np.square(actual - adv_images),
                                                                                    (1, 2, 3)))))
    mean_snr = np.mean(snr)
    print('snr :', mean_snr)
    
    max_index = np.argmax(prediction,1).reshape(-1,1)
    fix_indices = max_index * variance.shape[-1]    
    sigma_indices = max_index + fix_indices
    sigma_max = np.take_along_axis(variance.reshape(max_index.shape[0], -1), sigma_indices, axis=1)

    mean_var = np.mean(sigma_max).item()
    var_var = np.var(sigma_max).item()
    median_var = np.median(sigma_max).item()
    mad_var = stats.median_abs_deviation(sigma_max).item()
    mad_varf = np.median(np.absolute(sigma_max - median_var))
    
    print('Epsilon :', eps)
    print('mean var :', mean_var)
    print('var_var :', var_var)
    print('median_var :', median_var)
    print('mad_var :', mad_var)
    print('mad_varf :', mad_varf)
    
    cmax_index = np.argmax(corrmu,1).reshape(-1,1)
    cfix_indices = cmax_index * corrvar.shape[-1]    
    csigma_indices = cmax_index + cfix_indices
    csigma_max = np.take_along_axis(corrvar.reshape(cmax_index.shape[0], -1), csigma_indices, axis=1)
        
    cmedian_var = np.median(csigma_max).item()

    print('median_corrvar :', cmedian_var)
    print('mad_corrvar :', stats.median_abs_deviation(csigma_max).item())
    print('mad_corrvarf :', np.median(np.absolute(csigma_max - cmedian_var)))    
         
    imax_index = np.argmax(imu,1).reshape(-1,1)
    ifix_indices = imax_index * ivar.shape[-1]    
    isigma_indices = imax_index + ifix_indices
    isigma_max = np.take_along_axis(ivar.reshape(imax_index.shape[0], -1), isigma_indices, axis=1)
        
    imedian_var = np.median(isigma_max).item()
    print('median_ivar :', imedian_var)
    print('mad_ivar :', stats.median_abs_deviation(isigma_max).item())
    print('mad_ivarf :', np.median(np.absolute(isigma_max - imedian_var)))
    
    print('###################Threshold testing')    
#     threshold = [0.0008, 0.002, 0.03] #pathmnist     
#     threshold = [0.08, 0.2, 1] #dermamnist
    threshold = [2E-8, 1E-5, 2E-3] #organamnist
    for thresh in threshold:
        print('Threshold:', thresh)
        cpass = np.where(csigma_max <= thresh, 1, 0)
        corrpass = np.sum(cpass)
        cabs = np.where(csigma_max > thresh, 1, 0)
        corrabs = np.sum(cabs)
        ipass = np.where(isigma_max <= thresh, 1, 0)
        incorrpass = np.sum(ipass)
        iabs = np.where(isigma_max > thresh, 1, 0)
        incorrabs = np.sum(iabs)

        print('correct passed:', corrpass)
        print('incorrect passed:', incorrpass)
        print('correct abstained:', corrabs)
        print('incorrect abstained:', incorrabs)
        print('Abstained: ', (corrabs+incorrabs)/(corrpass+incorrpass+corrabs+incorrabs))
        print('Updated accuracy: ', corrpass/(corrpass+incorrpass))    
    
if __name__ == '__main__':
    main_run()