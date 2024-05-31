import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import os
from scipy.special import gamma



def estimate_GGD_parameters(vec):
    gam =np.arange(0.2,10.0,0.001)
    r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam))**2)
    sigma_sq=np.mean((vec)**2)
    sigma=np.sqrt(sigma_sq)
    E=np.mean(np.abs(vec-np.mean(vec)))
    r=sigma_sq/(E**2)
    diff=np.abs(r-r_gam)
    gamma_param=gam[np.argmin(diff, axis=0)]
    #print('Mean diff:', np.min(diff))
    return sigma, gamma_param
    
    
def KL_GGD3(sigma1, shape1, sigma2, shape2):
    I1 = shape1*sigma2*gamma(1/shape2)*np.sqrt(gamma(1/shape2)*gamma(3/shape1))
    I2 = shape2*sigma1*gamma(1/shape1)*np.sqrt(gamma(1/shape1)*gamma(3/shape2))
    I3 = sigma1*np.sqrt(gamma(1/shape1)*gamma(3/shape2))
    I4 = sigma2*np.sqrt(gamma(1/shape2)*gamma(3/shape1))
    
    A = np.log(I1/I2) - 1/shape1
    B = (I3/I4)**shape2 * (gamma(shape2/shape1+1/shape1)/gamma(1/shape1))
    out = A + B
    return out


# Assume that you have saved the deepest features (last layer) of the model
 
# test models
# PIES800_features32_MSRResNet_noGR_Train_DIV2K_clean
# PIES800_features32_MSRResNet_noGR_Train_DIV2K_blur0_4
# PIES800_features32_MSRResNet_noGR_Train_DIV2K_noise0_20
# PIES800_features32_MSRResNet_noGR_Train_DIV2K_blur2
# PIES800_features32_MSRResNet_noGR_Train_DIV2K_blur0_4_noise0_20
# PIES800_features32_DAN_noGR_setting1
# PIES800_features32_IKC_noGR

# PIES800_features32_RealESRGAN
# PIES800_features32_RealESRNet

# PIES800_features32_BSRGAN
# PIES800_features32_BSRNet

# PIES800_features32_DASR_iso_gaussian
# PIES800_features32_DASR_aniso_gaussian

# PIES800_features32_003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN
# PIES800_features32_003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR

model = 'PIES800_features32_MSRResNet_noGR_Train_DIV2K_clean' 
target_dataset = 'PIES-clean'

dataset2_list = ['PIES-clean', 'PIES-blur0.5', 'PIES-blur1', 'PIES-blur1.5', 'PIES-blur2', 'PIES-blur2.5', 'PIES-blur3', 
                  'PIES-blur3.5', 'PIES-blur4', 'PIES-blur4.5', 'PIES-blur5', 'PIES-blur5.5', 'PIES-blur6', 'PIES-blur6.5', 'PIES-blur7', 'PIES-blur7.5', 'PIES-blur8']

rfea_layer = 'HR_term'

PCA_dim = 300

anomaly_sigma = 5

data_volume = 800


kld_list = []
wd_list = []
count_dataset = 0
for dataset2 in [target_dataset]+dataset2_list:
    count_dataset += 1
    fea_npy_name2 = dataset2+'_'+rfea_layer+'.npy'  # eg., PIES-clean_HR_term.npy
    
    
    test_model = '{}_{}_{}_PCA{}'.format(model, dataset2, rfea_layer, PCA_dim)
    print('test SRGA model: ', test_model)
    
    b_path = '{}/{}'.format(model, fea_npy_name2) 
      
    b_name = dataset2 
    
    b = np.load(b_path)[0:data_volume]
    
    print('feature shape: ', b.shape)
    
    if 'DASR' in model:
       b = b/255.0 
    

    pca2 = PCA(n_components=PCA_dim, random_state=0)
    X2 = pca2.fit_transform(b)
    #print(np.sum(pca2.explained_variance_ratio_))
    
    X_tSNE = np.concatenate([X2],axis=0)
    print('after PCA: ', X_tSNE.shape)
    

    X2 = X_tSNE.reshape(-1)
    
    
    # remove anomalous data
    X2_mean, X2_std = np.mean(X2), np.std(X2)
    X2_list = list(X2)    
    count = 0
    
    
    for i in X2_list.copy():
        if i <= X2_mean - anomaly_sigma*np.std(X2) or i >= X2_mean + anomaly_sigma*np.std(X2):
            X2_list.remove(i)
            count += 1
    
    
    
    #print(len(X2_list))
    X2 = np.array(X2_list)
    
    sigma2, gamma_param2 = estimate_GGD_parameters(X2)
    #print('GGD:')
    print('sigma: {:.4f}  shape: {:.4f}'.format(sigma2, gamma_param2))
    
    
    if dataset2 == target_dataset:
        X_base = X2
        base_dataset = dataset2
        sigma_base = sigma2
        gamma_param_base = gamma_param2
    
    '''
    plt.subplot(2,4,count_dataset)
    plt.hist(X_base,bins=X2.shape[0]//5,alpha=0.5)
    plt.hist(X2,bins=X2.shape[0]//5,alpha=0.5)
    plt.legend([base_dataset, dataset2])
    
    #plt.show()
    '''
    
    kld = KL_GGD3(sigma1=sigma_base, shape1=gamma_param_base, sigma2=sigma2, shape2=gamma_param2)
    print('kld: {:.5f}'.format(kld))
    kld = np.log10(kld+10**(-5)) + 5
    kld_list.append(kld)
    print('SRGA (log kld): {:.5f}'.format(kld))
    




    