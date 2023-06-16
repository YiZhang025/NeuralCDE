import numpy as np 
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import os
from pathlib import Path
from scipy.io import loadmat 
from multiprocessing import Pool 
import time 
import collections

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from tqdm import tqdm
from itertools import repeat

from pymc import HalfCauchy, Model,sample, Normal, Uniform, InverseGamma, StudentT

import argparse


print(f"Running on PyMC v{pm.__version__}")

# from numba import cuda

def wall_clock_timing(func, *args, **kwargs):
    '''
    decorator that computes the wall clock running time 
    '''
    def timing_func(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"Time elapsed: {end - start:.3f} seconds")
        return ret 
    return timing_func


def T1_3param(t, C, k, T1):
    '''
    MOLLI signal acquistion equation s(t) = C(1 - k * np.exp(-t*(k-1)/T1))
    '''
    return C*(1 - k * np.exp(-t*(k-1)/T1))


def polarity_recovery(signal, tvec, p0=None, signal_threshold=25,
                      cache=None, cache_result=True,
                      **kwargs):
    '''
    Polarity recovery fitting, using cache can achieve an acceleration factor
    of ~2. 

    Parameters
    ----------
    signal : numpy.array
        signal acquisition array of shape (#acquisitions, )
    tvec : numpy.array
        inverstion time array of shape (#acquisitions, )
    signal_threshold : float
        threshold of maximal signal intensity, no protons will be assumed if max(signal) < threshold
    cache : List[dict]
        List of cached polarity recovery fitting results, for example results
        from the neignbouring voxels.
    cache_result : bool
        if the fitting intermediate results should be cached
    **kwargs 
        parameters that can be passed to curve fit
    
    '''
    # make safe code 
    signal = signal.squeeze().astype(np.float64)
    tvec = tvec.squeeze().astype(np.float64)
    
    # a few checks before running
    assert np.all(tvec[1:] - tvec[:-1] >= -1e-8), "Time vector is not sorted!"
    assert signal.shape == tvec.shape, "signal and tvec have mismatching shape!"
    
    # early exit 
    if signal.max() < signal_threshold: 
        return dict(p_opt=np.array([1., 0., 0.]), p_cov=None,
                    null_index=0, sd = 0, cache=None, sd_raw = 0, residuals = np.zeros(11))
    
    # adaptively initialize p0 
    if p0 is None:
        p0 = np.array([signal.max(), 2.0, 1e3])
    if cache is None:
        cache = [dict(p_opt=p0) for _ in tvec]
    assert len(cache) == tvec.shape[0], "cache has incorrect shape!"
    
    opt_results = []
    fitting_errors = [] 
    
    # polarity recovery 
    for null_index in range(signal.shape[-1]):
        signal_inverted = signal.copy()    #copy the array to avoid in-place change
        signal_inverted[:null_index] *= -1
        try:
            # ensure p0 is valid
            p0 = cache[null_index]["p_opt"]
            if kwargs.get("bounds") is not None:
                p0 = np.clip(p0, kwargs['bounds'][0], kwargs['bounds'][1])
            
            # optimize!
            p_opt, p_cov  = curve_fit(T1_3param, tvec, signal_inverted,
                                      p0=p0,
                                      **kwargs)
            
            # evaluate the mse error
            signal_fitted = T1_3param(tvec, *p_opt)
            mse = np.mean((signal_fitted - signal_inverted) ** 2)
        except Exception as e:
            print(cache[null_index]["p_opt"])
            print(kwargs)
            # handle the exception and indicate when the error is raised
            raise type(e)(str(e) + f"\nError raised when null_index={null_index:d}") 
       
        # save intermediate fitting results 
        opt_results.append(dict(p_opt=p_opt,
                                p_cov=p_cov,
                                mse=mse))
        fitting_errors.append(mse) 

    # find the best one
    best_null_index = np.argmin(fitting_errors) 
    best_opt_result = opt_results[best_null_index]
    best_opt_result["null_index"] = best_null_index

    signal_invert = signal.copy()
    signal_invert[:best_null_index] = -1*signal_invert[:best_null_index]
    best_opt_result['sd'], _  = sd_calculation(best_opt_result['p_opt'], tvec, signal_invert)
    _, best_opt_result['residuals'] = sd_calculation(best_opt_result['p_opt'], tvec, signal_invert)
    best_opt_result['sd_raw'] = np.sqrt(best_opt_result['p_cov'][2,2])

    # cache results if necessary 
    best_opt_result["cache"] = opt_results if cache_result else None 
    return best_opt_result


def split_volume(volume):
    '''
    build a generator that yields a row of volume at each call 
    
    Parameters
    ----------
    volume : numpy.array
        volume to be splited with shape [height, width, ...]
    '''
    for i in range(volume.shape[0]):
        yield volume[i, ...]


def fit_volume_row(signal_row, tvec, do_fitting_mask=None):
#     raise Exception("I am just crazy")
    '''
    fit a row of the volume
    
    Parameters
    ----------
    signal_row : numpy.array
        signal array with shape (width, #signal acquisitions)
    tvec : numpy.array
        inversion time vector with shape (#signal acquisitions, )
    do_fitting_mask : Union[None, numpy.array]
        per voxel tag indicating if fitting should be performed.
    '''
    if do_fitting_mask is None:
        do_fitting_mask = np.zeros(tvec.shape, dtype=bool)
    do_fitting_mask = np.squeeze(do_fitting_mask)
    
    # check the parameters first
    assert signal_row.shape[-1] == tvec.shape[-1], "Shape mismatch!"
    assert np.ndim(signal_row) == 2, "Please only pass a row of signal!"
    assert do_fitting_mask.shape == (signal_row.shape[0], ), "Fitting mask shape incorrect! " + str(do_fitting_mask.shape) 
    
    # initialize fitting result
    parameter_row = np.zeros((signal_row.shape[0], 3)) # 3 for A, B, T1*
    null_index_row = np.zeros((signal_row.shape[0], ))
    sd_row = np.zeros((signal_row.shape[0], ))

    # sort tvec
    order = np.argsort(tvec)
    tvec_sorted = tvec[order]
    signal_row_sorted = signal_row[..., order]
    
    # go through all voxels in a row
    cached = None
    for i in range(signal_row.shape[0]):
        if do_fitting_mask[i]:
            signal_sorted = signal_row_sorted[i, ...]
            # cached = fit_result["cache"] if i > 0  else None
            fit_result = polarity_recovery(signal_sorted, tvec_sorted,
                                           cache=cached,
                                           method="trf",
                                           bounds=(0, (signal_sorted.max() * 4., 15., 5e3)), # 0 < A < max(s) * 4, 0 < B < max(s) * 8, 0 < T1 < 5000
                                           max_nfev=5000)
            cached = fit_result["cache"]
            parameter_row[i, ...] = fit_result["p_opt"]
            null_index_row[i] = fit_result['null_index']
            sd_row[i] = fit_result["sd"]
        else:
            parameter_row[i, ...] = np.nan
            null_index_row[i] = np.nan
            sd_row[i] = np.nan
    return dict(p_opt=parameter_row, null_index=null_index_row, sd = sd_row)


#@wall_clock_timing
def polarity_recovery_volume(volume, tvec, do_fitting_mask=None, pool=None):
    '''
    Parallel fitting the whole volume,the row-wise fitting will take place in parallel
    if ``n_processes > 1``.
    
    Parameters
    ----------
    volume : numpy.array
        signal acquistion to be fitted of shape (height, width, #acquistions)
    tvec : numpy.array
        inversion time array of shape (#acquisitions,)
    do_fitting_mask : numpy.array
        bool array of shape (height, width) indicating if the fitting routine will be performed on a certain voxel
    pool : Pool 
        process pool for parallel fitting
    '''
    if do_fitting_mask is None:
        do_fitting_mask = np.ones(volume.shape[:2], dtype=bool)
        
    # check parameters
    assert np.ndim(volume) == 3, "Please pass volumes with shape (height, width, #acquisitions)"
    assert volume.shape[-1] == tvec.squeeze().shape[0], "Volume and tvec have mismatching shape!"
    assert do_fitting_mask.shape == volume.shape[:2], "do_fitting_mask has incorrect shape!"
        
    # start pool and run fitting per row
    if pool is None:
        print('start fitting.')
        # use single process only
        results_per_row = map(fit_volume_row, 
                              split_volume(volume),
                              (tvec for _ in range(volume.shape[0])),
                              split_volume(do_fitting_mask)
                             )
    else:
        # use multiple processes 
        results_per_row = pool.starmap(fit_volume_row,
                                       zip(split_volume(volume),
                                           (tvec for _ in range(volume.shape[0])),
                                           split_volume(do_fitting_mask)
                                          )
                                      )
    
    # assembly results per row
    parameter_estimated, null_index, sd = zip(*(
        (res["p_opt"], res["null_index"], res["sd"]) for res in results_per_row))
    parameter_estimated = np.stack(parameter_estimated, axis=0)
    null_index = np.stack(null_index, axis=0)
    sd = np.stack(sd, axis=0)
    
    return dict(p_opt=parameter_estimated, null_index=null_index, sd = sd)

    # define the t1 mapping model - this is for inverted data already
def model_t1(theta,x):
    # theta is composed of A, B, T1
    return theta[0] - theta[1]*np.exp(-1*x*(theta[1]/theta[0] - 1)/theta[2])


# define the derivative (Jacobian)
def jac(theta, x , y):
    J = np.empty((x.size, theta.size))
    term_share = np.exp(-x*(theta[1]/theta[0]-1)/theta[2])
    J[:, 0] = 1 - theta[1]*term_share*(x*theta[1]/(theta[2]*theta[0]**2))
    J[:, 1] = -1*term_share+theta[1]*term_share*(x/(theta[0]*theta[2]))
    J[:, 2] = -1*theta[1]*term_share*(x*(theta[1]/theta[0] -1 )/theta[2]**2)
    return J
    
# and Hessian approximation
def Hessian(jac, theta, x, y, mad):
  D = np.zeros([3,3])
  J = jac(theta,x,y)
  for i in range(D.shape[0]):
    for j in range(D.shape[1]):
      D[i,j] = np.multiply(J[:,i],J[:,j]).sum()
  return D/(mad**2)


def sd_calculation(p_opt, tvec, signal_invert):
  theta_opt = np.array([p_opt[0], p_opt[1]*p_opt[0], p_opt[2]])
  residuals = np.abs(model_t1(theta_opt, tvec)- signal_invert)
  residuals_raw = model_t1(theta_opt, tvec)- signal_invert
  residuals.sort()
  mad = np.median(residuals[2:])/0.6745
  D = Hessian(jac, theta_opt,tvec,signal_invert, mad)
  C = np.linalg.pinv(D)
  T1_sd_ls = np.sqrt(C[2,2])
  return T1_sd_ls, residuals_raw


# return per-pixel SD for MC iterative experiments - deprecated
def sd_estimation_pixel(t_vec, signal):
    res = polarity_recovery(signal, t_vec, p0=None, signal_threshold=25,
                      cache=None, cache_result=True, method="trf",
                                           bounds=(0, (signal.max() * 4., 15., 5e3)), # 0 < A < max(s) * 4, 0 < B < max(s) * 8, 0 < T1 < 5000
                                           max_nfev=5000)

    p_opt = res['p_opt']
    null_index = res['null_index']

    signal_invert = signal.copy()
    if not np.isnan(null_index):
        signal_invert[:null_index] = -1*signal[:null_index]

    return sd_calculation(p_opt, t_vec, signal_invert)

def loadmat_wrapper_Yi(data_path):
    '''
    A simple encapsulation of loadmat and data retrieval for MOLLI4Yi dataset - consider only v and tvec
    '''
    dat = loadmat(data_path) 
    # pmap_mse = dat["pmap_mse"]
    volume = dat["v"][160:285,145:270,1,:]
    tvec = dat["tvec"][1]
    #null_index = dat["null_index"]
    return dict(volume=volume,
                tvec=tvec,
                # pmap_mse=pmap_mse,
                # null_index=null_index
               )


def bayesian_fitting(t_vec, signal, sample_size = 1000, likelihood_func = 'Normal'):
    x_norm = t_vec/200
    y_norm = signal/100
    with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors, here T1 is also scaled down with the same mutiplier of x_norm
        T1_mu = Uniform('T1_mu', 1,10)
        sigma = InverseGamma("sigma", 0.001,0.001)
        C = Uniform("C", lower = 0.1, upper = 5)
        k = Uniform("k", lower = 1, upper = 3)
        T1 = Normal("T1", T1_mu, sigma = 0.2)
        
        
        # Define likelihood, with choices from Gaussian or Students' t distribution
        if likelihood_func == 'Normal':
            likelihood = Normal("y", mu=np.abs(C*(1 - k*np.exp(-1*x_norm*(k - 1)/T1))), sigma=sigma, observed=y_norm)
        if likelihood_func == 'StudentT':
            likelihood = StudentT("y", mu=np.abs(C*(1 - k*np.exp(-1*x_norm*(k - 1)/T1))), sigma=sigma, nu= 3, observed=y_norm)



        # Inference!
        # draw 2000 posterior samples using NUTS sampling
        idata = sample(chains = 2, cores = 1, tune = 1000, draws = 800,  progressbar= False)
        T1_mean = 200 * idata['posterior']['T1'].mean()
        T1_sd = 200* idata['posterior']['T1'].std()
        p_bayes = np.array ([200*idata['posterior']['T1'].mean(),100*idata['posterior']['C'].mean(),idata['posterior']['k'].mean()])

        return idata, T1_mean, T1_sd

def mc_experiment(ite, x, true_C, true_k, true_T1, size = 11):
    metric_to_save = []
    T1_estimate = []
    T1_sd_estimate = []
    T1_sd_raw_estimate = []
    T1_estimate_bayes = []
    T1_sd_estimate_bayes = []
    T1_estimate_bayes_Gauss = []
    T1_sd_estimate_bayes_Gauss = []

    fit_residuals = []

    for i in tqdm(range(ite)): 
        #$print(f'now on iteration {i+1}')
        #x = np.random.uniform(0,15,size)
        #x.sort()
        noise_x = np.random.uniform(-25,25,size).astype(int)
        x_perturb = x+noise_x
        true_regression_line = true_C*(1 - true_k*np.exp(-1*x*(true_k - 1)/true_T1))
        # add noise randomly
        y_perturb = true_regression_line + np.random.normal(scale=5, size=size)
        # or outlier - TO DO

        # calculate and save statistics
        res = polarity_recovery(y_perturb, x_perturb, p0=None, signal_threshold=25,
                            cache=None, cache_result=True, method="trf",
                                                bounds=(0, (y_perturb.max() * 4., 15., 5e3)), # 0 < A < max(s) * 4, 0 < B < max(s) * 8, 0 < T1 < 5000
                                                max_nfev=10000)

        _, t1, sd = bayesian_fitting(x_perturb,np.abs(y_perturb), likelihood_func = 'StudentT')
        _, t1_Gauss, sd_Gauss = bayesian_fitting(x_perturb,np.abs(y_perturb), likelihood_func = 'Normal')

        T1_estimate.append(res['p_opt'][2])
        T1_sd_estimate.append(res['sd'])
        T1_sd_raw_estimate.append(res['sd_raw'])
        T1_estimate_bayes.append(t1)
        T1_sd_estimate_bayes.append(sd)
        T1_estimate_bayes_Gauss.append(t1_Gauss)
        T1_sd_estimate_bayes_Gauss.append(sd_Gauss)
        fit_residuals.append(res['residuals'])
        
    T1_estimate = np.array(T1_estimate)
    T1_sd_estimate = np.array(T1_sd_estimate)
    T1_sd_raw_estimate = np.array(T1_sd_raw_estimate)
    T1_estimate_bayes = np.array(T1_estimate_bayes)
    T1_sd_estimate_bayes = np.array(T1_sd_estimate_bayes)
    T1_estimate_bayes_Gauss = np.array(T1_estimate_bayes_Gauss)
    T1_sd_estimate_bayes_Gauss = np.array(T1_sd_estimate_bayes_Gauss)
    
    fit_residuals = np.array(fit_residuals)
    residuals_resize = np.reshape(fit_residuals,(ite*11))

    #return dict(T1_estimate)
    
    return dict(T1_estimate = T1_estimate, T1_sd_estimate = T1_sd_estimate,T1_sd_raw_estimate = T1_sd_raw_estimate,
    T1_estimate_bayes = T1_estimate_bayes, T1_sd_estimate_bayes = T1_sd_estimate_bayes,
    T1_estimate_bayes_Gauss = T1_estimate_bayes_Gauss, T1_sd_estimate_bayes_Gauss =T1_sd_estimate_bayes_Gauss
    )
    
    
#def main():

if __name__ ==  '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--SNR', type = int)
    parser.add_argument('--k', default = 2, type = int)
    parser.add_argument('--T1', default= 1500, type = int)
    parser.add_argument('--trial_size', default = 2048, type = int)


    args = parser.parse_args()

    dat = loadmat('MAVI102_20151026_pre1.mat') 
    x_input = np.float64(dat['tvec'][2].copy())
    x_input.sort()

    # generate data with ground truth
    SNR = args.SNR
    size = 11
    true_C = 5*SNR
    true_k = args.k
    true_T1 = args.T1
    ite = args.trial_size
    
    n_cpu = os.cpu_count()
    print(f'now utilizing {n_cpu} cores')
    length_per_core = int(ite/n_cpu)
    # adding parallelism:
    with Pool(n_cpu) as pool:
    # calculate and save statistics
        results = pool.starmap(mc_experiment, zip([length_per_core]*n_cpu, repeat(x_input), repeat(true_C), repeat(true_k), repeat(true_T1),
        repeat(size)))
 
    print(results)
    
    dic = collections.defaultdict(list)

    for d in results:
        for k, v in d.items():
            dic[k].append(v)
    
    for i in dic:
        dic[i] = np.concatenate(dic[i])

    print (dic)


    #metric_to_save = [T1_estimate, T1_sd_estimate, T1_sd_raw_estimate, T1_estimate_bayes, T1_sd_estimate_bayes, T1_estimate_bayes_Gauss, T1_sd_estimate_bayes_Gauss, residuals_resize]
    np.save(f'metric_{args.T1}_{args.k}_{args.SNR}_{args.trial_size}.npy', dic)
