import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

def main_run():
    
    noise_std = 0
#     pf = open('../../Results/withNoise/PathMNIST_Res18/PathMNIST_Res18_Det_Speckle_Results_070322_2100_{}'.format(noise_std), 'rb')
#     pf = open('../../Results/withNoise/PathMNIST_Res18/PathMNIST_Res18_Det_Gaussian_Results_070322_2100_{}'.format(noise_std), 'rb')
    pf = open('../../Results/withNoise/DermaMNIST_Res18/DermaMNIST_Res18_Det_GN_Results_070322_1300_{}'.format(noise_std), 'rb')
#     pf = open('../../Results/withNoise/OrganaMNIST_Res18/OrganaMNIST_Res18_Det_GN_Results_240322_1200_{}'.format(noise_std), 'rb')
#     pf = open('../../Results/withNoise/OCTMNIST_Res18/OCTMNIST_Res18_Det_GN_Results_110422_1000_{}'.format(noise_std), 'rb')
    
    prediction, corrmu, imu = pickle.load(pf)
    pf.close()

    max_index = np.argmax(prediction,1).reshape(-1,1)

    pred_prob = np.take_along_axis(prediction, max_index, axis=1)
 
    
    mean_predprob = np.mean(pred_prob).item()
   
    print('noise std: ', noise_std)
    print('mean Predicted prob:', mean_predprob)
 
  
if __name__ == '__main__':
    main_run()