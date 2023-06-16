import wandb
import math
import torch
import torchcde
from tqdm import tqdm
import seaborn as sns
import numpy as np 
from matplotlib import pyplot as plt
import os
from pathlib import Path
from scipy.io import loadmat 
from multiprocessing import Pool 
import time 
from model import *
from data import *
import argparse
import random

#reproducibility
torch.manual_seed(1998)
np.random.seed(1998)
random.seed(1998)


sns.set_theme()

def train(model, train_dataloader, optimizer, loss_func, epoch, PINN = True, args = None):
    model.train()
    for batch in tqdm(train_dataloader):
        batch_coeffs, batch_y,batch_time = batch
        if args.cuda:
            batch_coeffs = batch_coeffs.cuda()
            batch_y = batch_y.cuda()
            batch_time = batch_time.cuda()

        optimizer.zero_grad()
        pred_y = model(batch_coeffs).squeeze(-1)

        # we instead use a linspace fit loss for the whole curve
        
        loss_PINN = 0
        if epoch > 2:
            if PINN:
                    loss_PINN = loss_func(pred_y*(torch.Tensor([100,1,1000]).cuda()), batch_y*(torch.Tensor([100,1,1000]).cuda()),torch.linspace(0,4000,30).cuda())
                    #print(loss_PINN)
                    wandb.log({"step_loss_PINN": loss_PINN})
                    
            else:
                loss_PINN = 0
        loss_T1_star = nn.functional.l1_loss(pred_y[:,2], batch_y[:,2])
        loss_C = nn.functional.l1_loss(pred_y[:,0], batch_y[:,0])
        loss_k = nn.functional.l1_loss(pred_y[:,1], batch_y[:,1])
        
        wandb.log({"step_loss_T1_star": loss_T1_star})
        wandb.log({"step_loss_C": loss_C})
        wandb.log({"step_loss_k": loss_k})

        loss_MAE = loss_T1_star + 10*loss_C + 1000*loss_k
        wandb.log({"step_loss_MAE": loss_MAE})
        loss = loss_MAE + 0.5*loss_PINN
        wandb.log({"step_loss": loss})

        loss.backward()
        optimizer.step()
        

    wandb.log({"epoch_loss": loss})

def validation(model, loss_func, test_dataloader, epoch, args = None):
    # validation is done by an iteration over set of images AND generated sequences

    model.eval()
    val_loss = []
    val_prediction = []
    for batch in tqdm(test_dataloader):

        if args.inverted:

            batch_coeffs, batch_y, = batch
            batch_coeffs = batch_coeffs.cuda()
            batch_y = batch_y.cuda()
            pred_y = model(batch_coeffs).squeeze(-1)
            # rescale pred_y to the original scale
            pred_y = pred_y*(torch.Tensor([100,1,1000]).cuda())
            loss_MAE = nn.functional.l1_loss(pred_y, batch_y)
            #loss_PINN = loss_func(pred_y, batch_y,torch.linspace(0,4000,100).cuda())
            #print(loss_PINN)
            loss = loss_MAE
            wandb.log({'step_loss_val_MAE': loss_MAE})
            #wandb.log({'step_loss_val_PINN': loss_PINN})
            wandb.log({"step_loss_val": loss})
            val_loss.append(loss.item())
            val_prediction.append(pred_y)

        else:
            # now we implement the polarity recovery via recursive search
            if args.model_type == 'NCDE':
                _, batch_y, batch_X, _ = batch
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()
                cached_loss = torch.zeros(batch_X.shape[1], device=batch_X.device)
                cached_results = torch.zeros(batch_X.shape[1], 3, device=batch_X.device)
                for i in range(batch_X.shape[1]):
                    X_recover = batch_X.clone()
                    X_recover[:,:i,1] = -batch_X[:,:i,1]
                    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_recover)
                    current_output = model(coeffs)
                    current_loss = curve_loss_val(current_output*(torch.Tensor([100,1,1000]).cuda()), batch_y*(torch.Tensor([100,1,1000]).cuda()), torch.linspace(100,4000,30).cuda())

                    if i == 0:
                        cached_loss = current_loss
                        cached_results = current_output
                    else:
                        cached_results = torch.where((current_loss < cached_loss).reshape(batch_X.shape[0],1).expand(-1,3), current_output, cached_results)
                        cached_loss = torch.where(current_loss < cached_loss, current_loss, cached_loss)
                    #print(cached_results*torch.Tensor([100,1,1000]).cuda())
                
                val_loss.append(cached_loss.mean().cpu().detach().numpy())
                val_prediction.append(cached_results)
            else:
                batch_coeffs, batch_y, = batch
                batch_coeffs = batch_coeffs.cuda()
                batch_y = batch_y.cuda()
                cached_loss = torch.zeros(batch_coeffs.shape[1], device=batch_coeffs.device)
                cached_results = torch.zeros(batch_coeffs.shape[1], 3, device=batch_coeffs.device)
                for i in range(batch_coeffs.shape[1]//2):
                    coeffs_recover = batch_coeffs.clone()
                    coeffs_recover[:,1:i*2+2:2] = -batch_coeffs[:,1:i*2+2:2]
                    current_output = model(coeffs_recover)
                    current_loss = curve_loss_val(current_output*(torch.Tensor([100,1,1000]).cuda()), batch_y*(torch.Tensor([100,1,1000]).cuda()), torch.linspace(100,4000,30).cuda())
                    if i == 0:
                        cached_loss = current_loss
                        cached_results = current_output
                    else:
                        cached_results = torch.where((current_loss < cached_loss).reshape(batch_coeffs.shape[0],1).expand(-1,3), current_output, cached_results)
                        cached_loss = torch.where(current_loss < cached_loss, current_loss, cached_loss)
                    #print(cached_results[4]*torch.Tensor([100,1,1000]).cuda())
                
                val_loss.append(cached_loss.mean().cpu().detach().numpy())
                val_prediction.append(cached_results)

    val_loss = np.array(val_loss)
    
    val_prediction_T1_star = torch.cat(val_prediction, dim=0)[:,2]
    val_prediction_k = torch.cat(val_prediction, dim=0)[:,1]
    val_prediction = val_prediction_T1_star*(val_prediction_k-1)
    val_prediction = val_prediction.cpu().detach().numpy()
    wandb.log({"epoch_loss_val": val_loss.mean(), "epoch": epoch})

    return val_prediction

def validation_image(model, loss_func, test_dataloader, epoch, args = None):
    # validation is done by an iteration over set of images AND generated sequences

    model.eval()
    val_loss = []
    val_prediction = []
    for batch in tqdm(test_dataloader):

        if args.inverted:

            batch_coeffs, batch_y, = batch
            batch_coeffs = batch_coeffs.cuda()
            batch_y = batch_y.cuda()
            pred_y = model(batch_coeffs).squeeze(-1)
            # rescale pred_y to the original scale
            # 
            pred_y = pred_y*(torch.Tensor([100,1,1000]).cuda())
            
            loss_MAE = nn.functional.l1_loss(pred_y, batch_y)
            #loss_PINN = loss_func(pred_y, batch_y,torch.linspace(0,4000,100).cuda())
            #print(loss_PINN)
            loss = loss_MAE
            wandb.log({'step_loss_val_MAE': loss_MAE})
            #wandb.log({'step_loss_val_PINN': loss_PINN})
            wandb.log({"step_loss_val": loss})
            val_loss.append(loss.item())
            val_prediction.append(pred_y)

        else:
            # now we implement the polarity recovery via recursive search
            if args.model_type == 'NCDE':
                _, batch_X, batch_y = batch
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()
                cached_loss = torch.zeros(batch_X.shape[1], device=batch_X.device)
                cached_results = torch.zeros(batch_X.shape[1], 3, device=batch_X.device)
                for i in range(batch_X.shape[1]):
                    X_recover = batch_X.clone()
                    X_recover[:,:i,1] = -batch_X[:,:i,1]
                    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_recover)
                    current_output = model(coeffs)
                    current_loss = curve_loss_val(current_output*(torch.Tensor([100,1,1000]).cuda()), batch_y*(torch.Tensor([100,1,1000]).cuda()), torch.linspace(100,4000,30).cuda())

                    if i == 0:
                        cached_loss = current_loss
                        cached_results = current_output
                    else:
                        cached_results = torch.where((current_loss < cached_loss).reshape(batch_X.shape[0],1).expand(-1,3), current_output, cached_results)
                        cached_loss = torch.where(current_loss < cached_loss, current_loss, cached_loss)
                    #print(cached_results*torch.Tensor([100,1,1000]).cuda())
                
                val_loss.append(cached_loss.mean().cpu().detach().numpy())
                val_prediction.append(cached_results*torch.Tensor([100,1,1000]).cuda())
            else:
                batch_coeffs, batch_y, = batch
                batch_coeffs = batch_coeffs.cuda()
                batch_y = batch_y.cuda()
                cached_loss = torch.zeros(batch_coeffs.shape[1], device=batch_coeffs.device)
                cached_results = torch.zeros(batch_coeffs.shape[1], 3, device=batch_coeffs.device)
                for i in range(batch_coeffs.shape[1]//2):
                    coeffs_recover = batch_coeffs.clone()
                    coeffs_recover[:,1:i*2+2:2] = -batch_coeffs[:,1:i*2+2:2]
                    current_output = model(coeffs_recover)
                    current_loss = curve_loss_val(current_output*(torch.Tensor([100,1,1000]).cuda()), batch_y*(torch.Tensor([100,1,1000]).cuda()), torch.linspace(100,4000,30).cuda())
                    if i == 0:
                        cached_loss = current_loss
                        cached_results = current_output
                    else:
                        cached_results = torch.where((current_loss < cached_loss).reshape(batch_coeffs.shape[0],1).expand(-1,3), current_output, cached_results)
                        cached_loss = torch.where(current_loss < cached_loss, current_loss, cached_loss)
                    print(cached_results*torch.Tensor([100,1,1000]).cuda())
                
                val_loss.append(cached_loss.mean().cpu().detach().numpy())
                val_prediction.append(cached_results*torch.Tensor([100,1,1000]).cuda())

    val_loss = np.array(val_loss)
    val_prediction_T1_star = torch.cat(val_prediction, dim=0)[:,2]
    val_prediction_k = torch.cat(val_prediction, dim=0)[:,1]
    val_prediction = val_prediction_T1_star*(val_prediction_k-1)
    val_prediction = val_prediction.cpu().detach().numpy()
    return val_prediction

def plot_t1_mapping(val_prediction, fitted_params, epoch):
    # plot 
    fig, (ax1,ax2) = plt.subplots(1,2, dpi=300)
    fig.suptitle(f'T1 Mapping at Epoch {epoch+1}', x = 0.56, y = 0.75)
    im = ax1.imshow(fitted_params.reshape(200,200,3)[:,:,2], vmin=200, vmax=2e3, cmap='jet')
    ax1.set_title('LS - T1 (ms)', y = -0.2)
    ax1.axis('off')

    plt.colorbar(im, fraction=0.040, pad=0.025)

    im = ax2.imshow(val_prediction.reshape(200,200), cmap='jet', vmin=200, vmax=2e3)
    ax2.axis('off')
    ax2.set_title('NCDE - T1 (ms)', y = -0.2)
    plt.colorbar(im, fraction=0.040, pad=0.025)

    plt.tight_layout()
    wandb.log({'validation_figure': fig})

def curve_loss_val(output, target,inv_times):
    target_signal = target[:,0].unsqueeze(-1)*(1 - target[:,1].unsqueeze(-1)*torch.exp(-1*inv_times.unsqueeze(0)/target[:,2].unsqueeze(-1)))
    output_signal = output[:,0].unsqueeze(-1)*(1 - output[:,1].unsqueeze(-1)*torch.exp(-1*inv_times.unsqueeze(0)/output[:,2].unsqueeze(-1)))
    #print(target_signal)
    #print(output_signal)
    loss = torch.abs(target_signal - output_signal)
    #print(loss)
    return loss.mean(axis=1)


def curve_loss(output, target,inv_times):
    target_signal = target[:,0].unsqueeze(-1)*(1 - target[:,1].unsqueeze(-1)*torch.exp(-1*inv_times.unsqueeze(0)/target[:,2].unsqueeze(-1)))
    output_signal = output[:,0].unsqueeze(-1)*(1 - output[:,1].unsqueeze(-1)*torch.exp(-1*inv_times.unsqueeze(0)/output[:,2].unsqueeze(-1)))
    #print(target_signal)
    #print(output_signal)
    loss = torch.mean(torch.abs(target_signal - output_signal))
    #print(loss)
    return loss



def main():
    parser = argparse.ArgumentParser()

    #Required parameters - to be filled in

    # hardware
    parser.add_argument("--cuda", default=True, type=bool, help="Use cuda or not.")

    # training
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train for.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")

    # loss function
    parser.add_argument('--PINN', default=False, type=bool, help="Use PINN loss or not.")
    
    # data
    parser.add_argument("--standarize", default=False, type=bool, help="Standarize the data or not.")
    parser.add_argument('--t_perturb', default=False, type=bool, help="Perturb the time axis or not.")
    parser.add_argument('--fixed', default=False, type=bool, help="Use fixed time axis or not.")
    parser.add_argument('--data_size', default=300000, type=int, help="Number of data points to generate.")
    parser.add_argument('--varying_length', default=False, type=bool, help="Use varying length or not.")
    parser.add_argument('--rician', default=False, type=bool, help="Use Rician noise or not.")
    parser.add_argument('--inverted',default=False, type=bool, help="Use inverted T1 recovery or not.")

    # model type
    parser.add_argument("--model_type", default='NCDE', type=str, help="Model type: NCDE or MLP.")

    # model parameters - NCDE-wise
    parser.add_argument("--input_channels", default=2, type=int, help="Number of input channels.")
    parser.add_argument("--hidden_channels", default=8, type=int, help="Number of hidden channels.")
    parser.add_argument("--output_channels", default=3, type=int, help="Number of output channels.")

    # Give arguments
    args = parser.parse_args()

    # Initialize wandb and log the config (hyperparameters)
    wandb.init(project="T1_mapping_with_polarity_recovery")

    wandb.config.update(args)
    #os.symlink(wandb.run.dir, wandb.run.dir.replace(wandb.run.id, wandb.run.name))
    # Generate training data

    if args.inverted:
        train_X, train_y, train_time = get_data_seq(dataset_size=args.data_size, fixed= args.fixed, t_perturb=args.t_perturb, varying_length=args.varying_length, rician=args.rician)
    else:
        train_X, train_y, train_time = get_data_seq_raw(dataset_size=args.data_size, fixed= args.fixed, t_perturb=args.t_perturb, varying_length=args.varying_length, rician=args.rician)

    # Generate validation data, the sizewill be 1/10 of the training data

    # to validate, always use inverted signals
    val_X, val_y, val_time = get_data_seq(dataset_size=int(args.data_size/10), fixed= args.fixed, t_perturb=args.t_perturb, varying_length=args.varying_length, rician=args.rician)

    # Normalize the data - either scaling directly or standarizing.

    train_X_norm = train_X.clone()
    train_y_norm = train_y.clone()
    train_time_norm = train_time.clone()

    val_X_norm = val_X.clone()
    val_y_norm = val_y.clone()
    val_time_norm = val_time.clone()

    if args.standarize:

        signal_mean = train_X[:,:,1].mean()
        signal_std = train_X[:,:,1].std()
        time_mean = train_X[:,:,0].mean()
        time_std = train_X[:,:,0].std()

        train_X_norm[:,:,1] = (train_X_norm[:,:,1] - signal_mean)/(signal_std+1e-5)
        train_X_norm[:,:,0] = (train_X_norm[:,:,0] - time_mean)/(time_std+1e-5)
        train_time_norm =  (train_time - train_time.mean())/(train_time.std()+1e-5)

    else:
        ### maybe add a parameter controlling the scale factor - now updated to 2000,200
        train_X_norm = train_X_norm/torch.Tensor([2000,200])
        train_time_norm = train_time_norm/torch.Tensor([2000])

        val_X_norm = val_X_norm/torch.Tensor([2000,200])
        val_time_norm = val_time_norm/torch.Tensor([2000])

    # In both cases, y is scaled by 1000,1,and 100
    train_y_norm = train_y_norm/torch.Tensor([100,1,1000])
    val_y_norm = val_y_norm/torch.Tensor([100,1,1000])

    # Now define the model, either NCDE or MLP
    if args.model_type == 'NCDE':
        model = NeuralCDE(input_channels=args.input_channels, hidden_channels=args.hidden_channels, output_channels=args.output_channels)
    elif args.model_type == 'MLP':
        # note here the meaning of channels are different from NCDE!
        model = MLP_myomapnet(input_channels=args.input_channels, hidden_channels=args.hidden_channels, output_channels=args.output_channels)
    else:
        raise ValueError('model type not supported yet, please choose from NCDE or MLP')
    
    # wheter cuda is used or not
    if args.cuda:
        model = model.cuda()
        train_X_norm = train_X_norm.cuda()
        train_y_norm = train_y_norm.cuda()
        train_time_norm = train_time_norm.cuda()

    # Optimizer
    if args.model_type == 'NCDE':
        if False:
            optimizer = torch.optim.Adam([{'params': model.readout.parameters(), 'lr': 5e-2}], lr=1e-4) 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == 'MLP':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # PINN loss - defined as curve_loss in the code
    
    # Data loader
    if  args.model_type == 'NCDE':
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X_norm)
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y_norm, train_time_norm)
        val_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(val_X_norm)
        val_dataset = torch.utils.data.TensorDataset(val_coeffs, val_y_norm, val_X_norm, val_time_norm)
    elif args.model_type == 'MLP':
        train_X_norm = (train_X_norm).reshape([-1,22])
        val_X_norm = (val_X_norm).reshape([-1,22])
        print(train_X_norm[0])
        train_dataset = torch.utils.data.TensorDataset(train_X_norm, train_y_norm, train_time_norm)
        val_dataset = torch.utils.data.TensorDataset(val_X_norm, val_y_norm)

    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)


    for epoch in tqdm(range(args.epochs)):
        print('current epoch: ', epoch, 'out of ', args.epochs, 'epochs')
        print(args.PINN)
        # Training and validation on the fly - use only one image for validation
        train(model = model, train_dataloader = train_dataloader,
               optimizer = optimizer, PINN = args.PINN, loss_func = curve_loss, epoch = epoch, args = args)
        

        # Validation, currently on only one image and reconstruct it after the 'fit'

        # The validation is also on generated sequences

        # Validation on the fly

        validation(model = model, test_dataloader = val_dataloader, loss_func = curve_loss_val, epoch = epoch, args=args)

        # test data loading and constructing test dataloader - to be updated later
        # this first line to be updated for a pytorch dataset
        data_Yi = loadmat_wrapper_Yi('/mnt/vol6t/Projects/NeuralCDE/T1_mapping/MOLLI4Yi/MAVI102_20151026_pre1.mat',1)
        data_tensor = torch.Tensor(data_Yi['volume'].astype(np.float32))
        time_tensor = torch.Tensor(data_Yi['tvec'].astype(np.float32))
        # always sort...
        time_tensor, index = torch.sort(time_tensor)
        data_tensor = data_tensor[:,:,index]
        data_reshaped = data_tensor.reshape(-1,11)
        fitted_params = torch.Tensor(np.load('/mnt/vol6t/Projects/NeuralCDE/T1_mapping/LS_results/MAVI102_20151026_pre1.mat_slice_1.npy'))
        fitted_params = fitted_params.reshape(-1, 3)
        test_y = fitted_params

        X_test = torch.stack([time_tensor.unsqueeze(0).repeat(data_tensor.shape[0]* data_tensor.shape[1], 1), data_reshaped], dim=2)
        X_test_norm = X_test.clone()

        if args.standarize:
            X_test_norm[:,:,1] = (X_test_norm[:,:,1] - signal_mean)/(signal_std+1e-5)
            X_test_norm[:,:,0] = (X_test_norm[:,:,0] - time_mean)/(time_std+1e-5)
        else:
            X_test_norm = X_test_norm/torch.Tensor([2000,200])
        #test_y_norm = fitted_params/torch.Tensor([100,1,1000])
        if args.model_type == 'NCDE':
            test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_test_norm)
            test_coeffs = test_coeffs.cuda()
            test_dataset = torch.utils.data.TensorDataset(test_coeffs, X_test_norm, test_y)
        elif args.model_type == 'MLP':
            X_test_norm = X_test_norm.reshape([-1,22])
            test_dataset = torch.utils.data.TensorDataset(X_test_norm.reshape([-1,22]), test_y)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048)
        
        test_y = test_y.cuda()
        val_prediction = validation_image(model = model, test_dataloader=test_dataloader ,loss_func= curve_loss, epoch = epoch, args = args)
        plot_t1_mapping(val_prediction, fitted_params, epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'model_at_epoch_{epoch+1}.pt'))


if __name__ == '__main__':
    main()