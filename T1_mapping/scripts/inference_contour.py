import wandb
import math
import torch
import glob
import torchcde
from tqdm import tqdm
import seaborn as sns
import numpy as np 
from matplotlib import pyplot as plt
import os
from pathlib import Path
from scipy.io import loadmat 
import time 
from model import *
from data import *

# Load Pretrained Model - 1: NCDE, 2: MLP
model = NeuralCDE(input_channels=2, hidden_channels=3, output_channels=3)
model = model.cuda()
model.load_state_dict(torch.load('/mnt/vol6t/Projects/NeuralCDE/T1_mapping/scripts/wandb/run-20230504_171827-94mjf1q9/files/model.pt'))

model_2 = MLP(input_channels=22, hidden_channels=128, output_channels=3)
model_2 = model_2.cuda()
model_2.load_state_dict(torch.load('/mnt/vol6t/Projects/NeuralCDE/T1_mapping/scripts/wandb/run-20230504_163728-7u5k60ut/files/model.pt'))

def get_data_stack(data):
    data_tensor = torch.Tensor(data['volume'].astype(np.float32))
    time_tensor = torch.Tensor(data['tvec'].astype(np.float32))
    time_tensor_unsorted = time_tensor.clone()
    # now add sorting
    time_tensor, index = torch.sort(time_tensor)
    data_tensor = data_tensor[:,:,index]

    data_reshaped = data_tensor.reshape(-1,11)
    X_test = torch.stack([time_tensor.unsqueeze(0).repeat(data_tensor.shape[0]* data_tensor.shape[1], 1), data_reshaped], dim=2)
    return X_test, time_tensor_unsorted, data_reshaped


# generate inference data numpy
data_root = '/mnt/vol6t/Projects/NeuralCDE/T1_mapping/registered_dataset'
data_files = glob.glob(data_root + '/*_org.pt')
for i in tqdm(range(len(data_files))):
    data = torch.load(data_files[i])
    i = data_files[i].split('/')[-1].split('.')[0]
    masked_T1 = np.ma.masked_array(data['T1'], mask = 1-data['mask'].astype('int'), fill_value= 0)
    X_test,time_tensor_unsorted, data_reshaped = get_data_stack(data)
    
    nonzero_indices = np.nonzero(data['T1'])
    top = np.min(nonzero_indices[0])
    bottom = np.max(nonzero_indices[0])
    left = np.min(nonzero_indices[1])
    right = np.max(nonzero_indices[1])
    height = bottom - top + 1
    width = right - left + 1

    # Compute the center crop dimensions
    crop_size = min(height, width)
    crop_top = top + (height - crop_size) // 2
    crop_bottom = crop_top + crop_size - 1
    crop_left = left + (width - crop_size) // 2
    crop_right = crop_left + crop_size - 1

    #X_test = torch.stack([time_tensor.unsqueeze(0).repeat(data_tensor.shape[0]* data_tensor.shape[1], 1), data_reshaped], dim=2)

    fitted_params = torch.Tensor(data['T1'])
    fitted_params = fitted_params.reshape(-1)
    ######################
    # Validation
    ######################
    test_X = X_test
    test_y = fitted_params
    test_X_norm = test_X/torch.Tensor([1000,100])
    #test_y_norm = test_y/torch.Tensor([100,1,1000])
    #test_time_norm = test_time/1000

    text_X_norm_mlp = test_X_norm.reshape(-1,22)
    print(text_X_norm_mlp.shape)
    model.eval()

    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X_norm)
    test_coeffs = test_coeffs.cuda()
    test_y = test_y.cuda()

    test_dataset = torch.utils.data.TensorDataset(test_coeffs, test_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8192)

    test_dataset_mlp = torch.utils.data.TensorDataset(text_X_norm_mlp.cuda(), test_y)
    test_dataloader_mlp = torch.utils.data.DataLoader(test_dataset_mlp, batch_size=8192)

    val_loss = []
    val_prediction = []
    for batch in tqdm(test_dataloader):
        batch_coeffs, batch_y, = batch
        batch_coeffs = batch_coeffs.cuda()
        batch_y = batch_y.cuda()
        pred_y = model(batch_coeffs).squeeze(-1)
        loss = nn.functional.l1_loss(pred_y[:,2], batch_y)
        # rescale pred_y to the original scale
        pred_y = pred_y*(torch.Tensor([100,1,1000]).cuda())
        val_prediction.append(pred_y)
        val_loss.append(loss.item())

    val_prediction = torch.cat(val_prediction, dim=0)[:,2]
    val_prediction = val_prediction.cpu().detach().numpy()
    val_loss = np.array(val_loss)
    print(val_loss.mean())

    val_loss = []
    val_prediction_mlp = []
    for batch in tqdm(test_dataloader_mlp):
        batch_x, batch_y, = batch
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        pred_y = model_2(batch_x).squeeze(-1)
        loss = nn.functional.l1_loss(pred_y[:,2], batch_y)
        # rescale pred_y to the original scale
        pred_y = pred_y*(torch.Tensor([100,1,1000]).cuda())
        val_prediction_mlp.append(pred_y)
        val_loss.append(loss.item())

    val_prediction_mlp = torch.cat(val_prediction_mlp, dim=0)[:,2]
    val_prediction_mlp = val_prediction_mlp.cpu().detach().numpy()
    val_loss = np.array(val_loss)
    print(val_loss.mean())

    # arrange mask for visualization and real evaluation

    val_prediction_mlp = val_prediction_mlp.reshape(data['T1'].shape[0],data['T1'].shape[1])
    masked_mlp = np.ma.masked_array(val_prediction_mlp, mask = 1-data['T1'].astype('bool'), fill_value= 0)[crop_top:crop_bottom+1, crop_left:crop_right+1]
    masked_mlp_myo = np.ma.masked_array(val_prediction_mlp, mask = 1-data['mask'].astype('int'), fill_value= 0)[crop_top:crop_bottom+1, crop_left:crop_right+1]

    val_prediction = val_prediction.reshape(data['T1'].shape[0],data['T1'].shape[1])
    masked_ncde = np.ma.masked_array(val_prediction, mask = 1-data['T1'].astype('bool'), fill_value= 0)[crop_top:crop_bottom+1, crop_left:crop_right+1]
    masked_ncde_myo = np.ma.masked_array(val_prediction, mask = 1-data['mask'].astype('int'), fill_value= 0)[crop_top:crop_bottom+1, crop_left:crop_right+1]

    masked_T1_myo = np.ma.masked_array(data['T1'], mask = 1-data['mask'].astype('int'), fill_value= 0)[crop_top:crop_bottom+1, crop_left:crop_right+1]

    diff_myo_ls_mlp = abs(masked_mlp_myo - masked_T1_myo)
    diff_myo_ls_ncde = abs(masked_ncde_myo - masked_T1_myo)
    T1_ls_mean = masked_T1_myo.mean()
    T1_mlp_mean = masked_mlp_myo.mean()
    T1_ncde_mean = masked_ncde_myo.mean()
    T1_ls_sd = masked_T1_myo.std()
    T1_mlp_sd = masked_mlp_myo.std()
    T1_ncde_sd = masked_ncde_myo.std()
    print(diff_myo_ls_mlp.mean(), diff_myo_ls_ncde.mean())
    print(T1_ls_mean, T1_mlp_mean, T1_ncde_mean)
    print(T1_ls_sd, T1_mlp_sd, T1_ncde_sd)
    torch.save([diff_myo_ls_mlp.mean(), diff_myo_ls_ncde.mean(), T1_ls_mean, T1_mlp_mean, T1_ncde_mean, T1_ls_sd, T1_mlp_sd, T1_ncde_sd]
               , f'inference_results_org/T1_mapping_metrics/{i}_metrics.pt')
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, dpi=300)
    fig.suptitle(f'T1 Mapping - Myocardium Only', y = 0.67)
    im = ax1.imshow(masked_T1_myo.filled(0), vmin=200, vmax=2e3, cmap='jet')
    ax1.set_title(f'LS - T1 (ms) \n mean = {masked_T1_myo.mean():.2f}', y = -0.3, fontsize = 9)
    ax1.axis('off')

    plt.colorbar(im, fraction=0.045, pad=0.025)

    im = ax2.imshow(masked_mlp_myo.filled(0), cmap='jet', vmin=200, vmax=2e3)
    ax2.axis('off')
    ax2.set_title(f'MLP - T1 (ms) \n mean = {masked_mlp_myo.mean():.2f}', y = -0.3, fontsize = 9)
    plt.colorbar(im, fraction=0.045, pad=0.025)

    im = ax3.imshow(masked_ncde_myo.filled(0), cmap='jet', vmin=200, vmax=2e3)
    ax3.axis('off')
    ax3.set_title(f'NCDE - T1 (ms) \n mean = {masked_ncde_myo.mean():.2f}', y = -0.3, fontsize = 9)
    plt.colorbar(im, fraction=0.045, pad=0.025)

    fig.tight_layout()
    plt.savefig(f'inference_results_org/T1_mapping_myo/{i}_myo.png')

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, dpi=300)
    fig.suptitle(f'T1 Mapping', y = 0.65)
    im = ax1.imshow(data['T1'][crop_top:crop_bottom+1, crop_left:crop_right+1], vmin=200, vmax=2e3, cmap='jet')
    x = [point[0]-crop_left for point in data['outer_contour']]
    y = [point[1]-crop_top for point in data['outer_contour']]

    # Plot the contour
    im1 = ax1.plot(x, y,color = 'green')

    x = [point[0]-crop_left for point in data['inner_contour']]
    y = [point[1]-crop_top for point in data['inner_contour']]

    # Plot the contour
    im2 = ax1.plot(x, y, color = 'red')
    ax1.set_title('LS - T1 (ms)', y = -0.2)
    ax1.axis('off')

    plt.colorbar(im, fraction=0.040, pad=0.025)

    im = ax2.imshow(masked_mlp.filled(0), cmap='jet', vmin=200, vmax=2e3)
    ax2.axis('off')
    ax2.set_title('MLP - T1 (ms)', y = -0.2)
    plt.colorbar(im, fraction=0.040, pad=0.025)

    im = ax3.imshow(masked_ncde.filled(0), cmap='jet', vmin=200, vmax=2e3)
    ax3.axis('off')
    ax3.set_title('NCDE - T1 (ms)', y = -0.2)
    plt.colorbar(im, fraction=0.040, pad=0.025)

    fig.tight_layout()
    plt.savefig(f'inference_results_org/T1_mapping/{i}_T1.png')

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, dpi=300)
    fig.suptitle(f'T1 Mapping Difference - Myocardium Only \n', y = 0.65)
    im = ax1.imshow(masked_T1_myo.filled(0), vmin=200, vmax=2e3, cmap='jet')
    ax1.set_title(f'LS - T1 (ms) \n mean = {masked_T1_myo.mean():.2f}', y = -0.3, fontsize = 9)
    ax1.axis('off')

    plt.colorbar(im, fraction=0.045, pad=0.025)

    im = ax2.imshow(masked_T1_myo.filled(0) - masked_mlp_myo.filled(0), cmap='RdBu', vmin=-200, vmax=200)
    ax2.axis('off')
    ax2.set_title(f'(MLP - LS) - T1 (ms) \n MAE = {diff_myo_ls_mlp.mean():.2f}', y = -0.3, fontsize = 9)
    plt.colorbar(im, fraction=0.045, pad=0.025)

    im = ax3.imshow(masked_T1_myo.filled(0) - masked_ncde_myo.filled(0), cmap='RdBu', vmin=-200, vmax=200)
    ax3.axis('off')
    ax3.set_title(f'(NCDE - LS) - T1 (ms) \n MAE = {diff_myo_ls_ncde.mean():.2f}', y = -0.3, fontsize = 9)
    plt.colorbar(im, fraction=0.045, pad=0.025)


    fig.tight_layout()
    plt.savefig(f'inference_results_org/T1_mapping_diff/{i}_T1.png')