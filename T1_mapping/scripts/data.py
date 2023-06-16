from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from tqdm import tqdm
import torch
import scipy.stats as st
import scipy
import numpy as np
import matplotlib.pyplot as plt

######################
# Deprecated
######################
def loadmat_wrapper_Yi_old(data_path): 
    '''
    A simple encapsulation of loadmat and data retrieval for MOLLI4Yi dataset - consider only v and tvec
    '''
    dat = loadmat(data_path) 
    # pmap_mse = dat["pmap_mse"]
    volume = dat["v"][160:285,145:270,1,:]
    tvec = dat["tvec"][0]
    #null_index = dat["null_index"]
    return dict(volume=volume,
                tvec=tvec,
                # pmap_mse=pmap_mse,
                # null_index=null_index
               )

def loadmat_wrapper_Yi(data_path, slice_index): 
    '''
    A simple encapsulation of loadmat and data retrieval for MOLLI4Yi dataset - consider only v and tvec
    '''
    dat = loadmat(data_path) 
    # pmap_mse = dat["pmap_mse"]
    volume = dat["v"][100:300,100:300,slice_index,:]
    tvec = dat["tvec"][slice_index]
    #null_index = dat["null_index"]
    return dict(volume=volume,
                tvec=tvec,
                # pmap_mse=pmap_mse,
                # null_index=null_index
               )

def loadmat_wrapper_with_contour(data_path, slice_index): 
    '''
    A simple encapsulation of loadmat and data retrieval for the T1 mapping dataset with T1, sd, contour now
    '''
    dat = loadmat(data_path) 
    # pmap_mse = dat["pmap_mse"]

    try:
        volume = dat["volume"][:,:,slice_index,:]
        tvec = dat["tvec"][slice_index]
    except IndexError:
        print('invalid slice index, return None')
        return dict(volume=None,
                    tvec=None,
                    T1 = None,
                    sd_t = None,
                    outer_contour = None,
                    inner_contour = None,
                    mask = None)

    
    if dat['volume'].shape[2] == 1:
        T1 = dat["T1"]
        sd_t = dat["sd_t"]
    else:
        T1 = dat["T1"][:,:,slice_index]
        try:
            sd_t = dat["sd_t"][:,:,slice_index]
        except KeyError:
            sd_t = None
    try:
        outer_contour = dat["contour2"][slice_index,0][0][0][1]
        inner_contour = dat["contour2"][slice_index,0][0][0][0]

        outer_path = Path(outer_contour)
        inner_path = Path(inner_contour)
        
        x = np.arange(0, volume.shape[0], 1)
        y = np.arange(0, volume.shape[1], 1)
        xx, yy = np.meshgrid(x, y)
        mask = np.zeros_like(xx)

        points = np.column_stack([xx.ravel(), yy.ravel()])
        outer_mask = outer_path.contains_points(points)
        inner_mask = inner_path.contains_points(points)
        mask = outer_mask & (~inner_mask)
        mask = mask.reshape(xx.shape)

    except ValueError:
        return dict(volume=volume,
                    tvec=tvec,
                    T1 = T1,
                    sd_t = sd_t,
                    outer_contour = None,
                    inner_contour = None,
                    mask = None)

    #null_index = dat["null_index"]
    return dict(volume=volume,
                tvec=tvec,
                T1 = T1,
                sd_t = sd_t,
                outer_contour = outer_contour,
                inner_contour = inner_contour,
                mask = mask
                # pmap_mse=pmap_mse,
                # null_index=null_index
               )


def loadmat_wrapper_with_contour_org(data_path, slice_index): 
    '''
    A simple encapsulation of loadmat and data retrieval for the T1 mapping dataset with T1, sd, contour now
    '''
    dat = loadmat(data_path) 
    # pmap_mse = dat["pmap_mse"]

    try:
        volume = dat["volume"][:,:,slice_index,:]
        tvec = dat["tvec"][slice_index]
    except IndexError:
        print('invalid slice index, return None')
        return dict(volume=None,
                    tvec=None,
                    T1 = None,
                    sd_t = None,
                    outer_contour = None,
                    inner_contour = None,
                    mask = None)

    
    if dat['volume'].shape[2] == 1:
        T1 = dat["T1"]
        sd_t = dat["sd_t"]
    else:
        T1 = dat["T1"][:,:,slice_index]
        try:
            sd_t = dat["sd_t"][:,:,slice_index]
        except KeyError:
            sd_t = None
    try:
        outer_contour = dat["contour2_org"][slice_index,0][0][0][1]
        inner_contour = dat["contour2_org"][slice_index,0][0][0][0]

        outer_path = Path(outer_contour)
        inner_path = Path(inner_contour)
        
        x = np.arange(0, volume.shape[0], 1)
        y = np.arange(0, volume.shape[1], 1)
        xx, yy = np.meshgrid(x, y)
        mask = np.zeros_like(xx)

        points = np.column_stack([xx.ravel(), yy.ravel()])
        outer_mask = outer_path.contains_points(points)
        inner_mask = inner_path.contains_points(points)
        mask = outer_mask & (~inner_mask)
        mask = mask.reshape(xx.shape)

    except ValueError:
        return dict(volume=volume,
                    tvec=tvec,
                    T1 = T1,
                    sd_t = sd_t,
                    outer_contour = None,
                    inner_contour = None,
                    mask = None)

    #null_index = dat["null_index"]
    return dict(volume=volume,
                tvec=tvec,
                T1 = T1,
                sd_t = sd_t,
                outer_contour = outer_contour,
                inner_contour = inner_contour,
                mask = mask 
                # pmap_mse=pmap_mse,
                # null_index=null_index
               )

def get_data_seq(num_timepoints=11, fixed = True, t_perturb = False, varying_length = False, dataset_size = 65536, rician = False):
    
    ######################
    # choose a fixed time sequence or a random one
    # The fixed sequence is given from the orignal data to be tested...
    ######################

    if fixed:
        #t,_ = torch.tensor([ 114., 1043, 1991,  232, 1187, 2145,  350, 1301, 2245, 3211, 4190]).sort()
        t= torch.tensor([ 137., 1029.0, 1940.0,  243.0, 1165.0, 2043.0,  350.0, 1307.0, 2211.0, 3124.0, 4056.0])
        # always sort the time sequence
        t,_ = torch.sort(t)
    else:
        t = torch.linspace(0., 4200., num_timepoints)

    ######################
    # Define a range of synthetic curves
    # with c, k, T1 as parameters
    # the data is generated from the UNIFORM distribution over predefined ranges.
    # C in [50, 400], k in [1, 4], T1 in [100, 2600], with absolute value of the signal. 
    # Assume SNR in [50, 100]
    ######################   
    C = (500-10) * torch.rand(dataset_size,1) + 10
    k = (4-1.2) * torch.rand(dataset_size,1) + 1.2
    T1_star = (2600-100) * torch.rand(dataset_size,1) + 100
    SNR = (100-15) * torch.rand(dataset_size,1) + 15
    
    #print(f'SNR is {SNR}, magnitude of noise is {C/SNR}')
    # Define rician noise distribution
    class RicianPDF(st.rv_continuous):
        def _pdf(self, s, real_intensity, sigma):
            self.real_intensity = real_intensity
            self.sigma = sigma
            return (s/sigma**2) * np.exp(-(s**2 + real_intensity**2)/(2*sigma**2)) * scipy.special.i0(s*real_intensity/sigma**2)
    
    # the support of rician distribution is (0,\infty)
    rician_noise = RicianPDF(a=0, name='RicianPDF')    

    # what if we add noise to time?
    if t_perturb:
        X = torch.zeros(dataset_size, len(t),2)
        for i in tqdm(range(dataset_size)):
            # TODO: write in vectorized form
            perturbed_t = (t + torch.rand(len(t))*400 - 200).sort()[0]
            if varying_length:
                #perm = torch.randperm(perturbed_t.size(0))
                #perm = perm[:torch.randint(11,11,(1,))]
                #t_sampled = perturbed_t[perm].sort()[0]
                t_sampled = perturbed_t
                if rician:
                    signal_sampled = torch.abs(C[i]*(1 - k[i]*torch.exp(-1*(t_sampled).unsqueeze(0)/T1_star[i])))
                    n_level = (C[i]/SNR[i])
                    for j in range(len(t_sampled)):
                        
                        if (signal_sampled[0,j] < n_level) and (signal_sampled[0,j] > 0):
                            #print('rician noise applied!')
                            #print(signal_sampled[0,j])
                            signal_sampled[0,j] = abs(signal_sampled[0,j] + rician_noise.rvs(real_intensity = signal_sampled[0,j], sigma = C[i]/SNR[i]))
                        else:
                            signal_sampled[0,j] = abs(signal_sampled[0,j] + torch.randn(1)* (C[i] / SNR[i]))

                        
                else:                
                    signal_sampled = torch.abs(C[i]*(1 - k[i]*torch.exp(-1*(t_sampled).unsqueeze(0)/T1_star[i]) + torch.randn(len(t_sampled))* (C[i] / SNR[i])))
                signal_sampled = signal_sampled.squeeze(0)
                x = torch.stack([t_sampled, signal_sampled], dim=1)
                x = torch.cat([x, x[-1].unsqueeze(0).expand(11 - x.size(0), x.size(1))])
                X[i] = x

            else:
                X[i,:,0] = perturbed_t
                X[i,:,1] = torch.abs(C[i]*(1 - k[i]*torch.exp(-1*(t).unsqueeze(0)/T1_star[i])) + torch.randn(len(t))* (C[i] / SNR[i]))
            
    else:
        X = torch.abs(C*(1 - k*torch.exp(-1*(t).unsqueeze(0)/T1_star))) + torch.randn(dataset_size, len(t))* (C / SNR)
        X = torch.stack([t.unsqueeze(0).repeat(dataset_size, 1), X], dim=2)
    y = torch.concat([C, k, T1_star], dim=1)

    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    ######################

    ######################
    # We can also create a dataset of which the length of each data point is different
    ######################



    # permuting the order of dataset
    perm = torch.randperm(dataset_size)

    X = X[perm]
    y = y[perm]
    time = X[perm,:,0]
    #time = time[perm]
    ######################
    # X is a tensor of observations, of shape (batch, sequence=len(t), channels=1)
    # y is a tensor of underlined parameters, of shape (batch, 3)
    # respectively.
    ######################
    return X, y, time

def get_data_seq_raw(num_timepoints=200, fixed = True, t_perturb = False, varying_length = False, dataset_size = 65536, rician = False):
    
    ######################
    # choose a fixed time sequence or a random one
    # The fixed sequence is given from the orignal data to be tested...
    ######################

    if fixed:
        #t,_ = torch.tensor([ 114., 1043, 1991,  232, 1187, 2145,  350, 1301, 2245, 3211, 4190]).sort()
        t= torch.tensor([ 137., 1029.0, 1940.0,  243.0, 1165.0, 2043.0,  350.0, 1307.0, 2211.0, 3124.0, 4056.0])
        # always sort the time sequence
        t,_ = torch.sort(t)
    else:
        t = torch.linspace(0., 4200., num_timepoints)

    ######################
    # Define a range of synthetic curves
    # with c, k, T1 as parameters
    # the data is generated from the UNIFORM distribution over predefined ranges.
    # C in [50, 400], k in [1.2, 4], T1_star in [100, 2600], with absolute value of the signal. 
    # Assume SNR in [50, 100]
    ######################   
    C = (500-10) * torch.rand(dataset_size,1) + 10
    k = (4-1.2) * torch.rand(dataset_size,1) + 1.2
    T1_star = (2600-100) * torch.rand(dataset_size,1) + 100
    SNR = (100-15) * torch.rand(dataset_size,1) + 15
    
    #print(f'SNR is {SNR}, magnitude of noise is {C/SNR}')
    # Define rician noise distribution
    class RicianPDF(st.rv_continuous):
        def _pdf(self, s, real_intensity, sigma):
            self.real_intensity = real_intensity
            self.sigma = sigma
            return (s/sigma**2) * np.exp(-(s**2 + real_intensity**2)/(2*sigma**2)) * scipy.special.i0(s*real_intensity/sigma**2)
    
    # the support of rician distribution is (0,\infty)
    rician_noise = RicianPDF(a=0, name='RicianPDF')    

    # what if we add noise to time?
    if t_perturb:
        X = torch.zeros(dataset_size, len(t),2)
        for i in tqdm(range(dataset_size)):
            # TODO: write in vectorized form
            # perturb should not be used simultaneously with random interval
            perturbed_t = (t + torch.rand(len(t))*400 - 200).sort()[0]
            if varying_length:
                # when the length is varying...
                #perm = torch.randperm(perturbed_t.size(0))
                #perm = perm[:torch.randint(11,11,(1,))]
                #t_sampled = perturbed_t[perm].sort()[0]
                t_sampled = perturbed_t
                # when there's Rician
                if rician:
                    signal_sampled = C[i]*(1 - k[i]*torch.exp(-1*(t_sampled).unsqueeze(0)/T1_star[i]))
                    n_level = (C[i]/SNR[i])
                    for j in range(len(t_sampled)):
                        
                        if (signal_sampled[0,j] < n_level) and (signal_sampled[0,j] > 0):
                            #print('rician noise applied!')
                            #print(signal_sampled[0,j])
                            signal_sampled[0,j] = signal_sampled[0,j] + rician_noise.rvs(real_intensity = signal_sampled[0,j], sigma = C[i]/SNR[i])
                        else:
                            signal_sampled[0,j] = signal_sampled[0,j] + torch.randn(1)* (C[i] / SNR[i])

                        
                else:                
                    signal_sampled = C[i]*(1 - k[i]*torch.exp(-1*(t_sampled).unsqueeze(0)/T1_star[i])) + torch.randn(len(t_sampled))* (C[i] / SNR[i])
                signal_sampled = signal_sampled.squeeze(0)
                x = torch.stack([t_sampled, signal_sampled], dim=1)
                x = torch.cat([x, x[-1].unsqueeze(0).expand(11 - x.size(0), x.size(1))])
                X[i] = x

            else:
                X[i,:,0] = perturbed_t
                X[i,:,1] = C[i]*(1 - k[i]*torch.exp(-1*(t).unsqueeze(0)/T1_star[i])) + torch.randn(len(t))* (C[i] / SNR[i])            
            
    else:
        X = C*(1 - k*torch.exp(-1*(t).unsqueeze(0)/T1_star)) + torch.randn(dataset_size, len(t))* (C / SNR)
        X = torch.stack([t.unsqueeze(0).repeat(dataset_size, 1), X], dim=2)
    y = torch.concat([C, k, T1_star], dim=1)

    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    ######################

    ######################
    # We can also create a dataset of which the length of each data point is different
    ######################



    # permuting the order of dataset
    perm = torch.randperm(dataset_size)

    X = X[perm]
    y = y[perm]
    time = X[perm,:,0]
    #time = time[perm]
    ######################
    # X is a tensor of observations, of shape (batch, sequence=len(t), channels=1)
    # y is a tensor of underlined parameters, of shape (batch, 3)
    # respectively.
    ######################
    return X, y, time
