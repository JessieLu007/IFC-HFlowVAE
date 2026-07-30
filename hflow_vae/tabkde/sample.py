import os

import torch

import argparse
import warnings
import time

import pandas as pd
from tabkde.model import MLPDiffusion, Model
from tabkde.latent_utils import get_input_generate, get_input_train, recover_data, split_num_cat_target, convert_sample_z_to_df
from tabkde.diffusion_utils import sample


from tabkde.tools_1 import cprint

from tabkde.tools_1 import * 
from tabkde.tools import *

warnings.filterwarnings('ignore')
decorrelation = False
sigma = 'global'

def main(args):
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    parent_dir = os.path.dirname(base_dir)
 
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    if  args.method in ['simple_KDE', 'TabKDE']:
        # is {args.method}, args.latent_encoding has been changed to copula]', bold = True, color = 'red')
        args.latent_encoding = 'copula_encoding'
 

    if args.latent_encoding == 'vae':
        #print('latent encoding is based on VAE')
        train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    elif args.latent_encoding == 'copula_diff_encoding':
        #print('latent encoding is based on ordinal_encoding')
        train_z, curr_dir, dataset_dir, ckpt_path, info = get_input_train(args)
    else:
        if args.latent_encoding != 'copula_encoding':
            #cprint(f'Warning: latent_encoding should be copula and it is {args.latent_encoding}', bold = True, color = 'red')
            raise ValueError(f"Incompatible value for latent_encoding: {args.latent_encoding}.")
        train_z, curr_dir, dataset_dir, ckpt_path, info = get_input_train(args)

    # check method
    if args.method == 'tabsyn':
        in_dim = train_z.shape[1] 
        mean = train_z.mean(0)
        #print(f'ckpt_path is {ckpt_path}')

        denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
        
        model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    
        model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))
        #cprint(f'Diffusion model for {args.method} has been loaded form {ckpt_path}', color= 'green', bold = True)
        '''
            Generating samples    
        '''
        start_time = time.time()
    
        num_samples = args.num_samples#train_z.shape[0]
        sample_dim = in_dim
    
        x_next = sample(net = model.denoise_fn_D, num_samples = num_samples, dim = sample_dim, device = device)
        x_next = x_next * 2 + mean.to(device)
        syn_data = x_next.float().cpu().numpy()

        if args.latent_encoding == 'vae':
            #print(f'methood is {args.method} with latent_encoding {args.latent_encoding}')
            syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 
        
            syn_df = recover_data(syn_num, syn_cat, syn_target, info)
        
            idx_name_mapping = info['idx_name_mapping']
            idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
        
            syn_df.rename(columns = idx_name_mapping, inplace=True)
            syn_df.to_csv(save_path, index = False)
        elif args.latent_encoding == 'copula_diff_encoding':

            save_dir = os.path.join(parent_dir, "synthetic", dataname)
            os.makedirs(save_dir, exist_ok=True)  # make directory if it doesn't exist
            save_path = os.path.join(save_dir, "copulaDiff.csv")
            
            #print(f'methood is {args.method} with latent_encoding {args.latent_encoding} and save_path is {save_path}')
          
            syn_df = convert_sample_z_to_df(syn_data, args)
            syn_df.to_csv(save_path, index = False)
        
        end_time = time.time()
        #print('Time:', end_time - start_time)
    
        #print('Saving sampled data to {}'.format(save_path))
        return syn_df
    elif  args.method in ['simple_KDE_VAE_encoding','KDE_VAE_encoding']:
     
        #print(f'The generation method is {args.method}') 
        
        save_dir = os.path.join(parent_dir, "synthetic", dataname)
        os.makedirs(save_dir, exist_ok=True)  # make directory if it doesn't exist
        save_path = os.path.join(save_dir, f'{args.method}.csv')   
        #cprint(f'synthetic data will be saved on {save_path}', bold = True)
     
        '''
            Generating samples    
        '''
        start_time = time.time()
        
        npy_z = train_z.reshape(train_z.shape[0], -1)
        npy_z.shape
        df = pd.DataFrame(npy_z)
        
        model = EmpiricalTransformer(df)
        df_train_copula = model.fit(method = 'average') 

        #print('data has been transformed to Copula setting')

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = f'{curr_dir}/ckpt/{dataname}' 

        
        if args.dataname == 'ibm_func':
            random_selection_size = 15000
        elif args.dataname == 'news':
            random_selection_size = 10000
        else:
            random_selection_size = None
            
        #if random_selection_size: 
            #cprint('!We are in VAE latent space. So the simention is 4 times bigger!! Hard to compute the siatance',
                  #color = 'red', bold = True)
            #cprint(f'As data is {args.dataname}, for KDTree a random subselection of size {random_selection_size} is used.', color = 'green')
        
        
        gmm_BIC, gmm_AIC , DCP_data = fit_mix_gaussian_to_(df_train_copula, 
                                                           partition_portion = .5, 
                                                           max_number_components = 10,
                                                           saved_path_for_plot=f'{ckpt_dir}/mix_gaussian.png', 
                                                           random_selection_size = random_selection_size,
                                                           down_sampling = 200000 if args.dataname in ['ibm_func', 'news'] else None
                                                          )
        #print('mix_gaussian has been fitted to the DCR dist.')
        n, d = df_train_copula.shape

        X = df_train_copula.sample(n=n, random_state=None, replace = True).values
        
        X_prime = perturb_vector_with_partial_resampling_with_DCP(X, 
                                                                  model = gmm_BIC, 
                                                                  num_samples = args.num_samples,
                                                                  sigma = sigma, 
                                                                  allowed_outside_boundary = False if args.method =='KDE_VAE_encoding' else True)
        synth_data = model.convert(X_prime).to_numpy()
     
        synth_data_pre_vae_decodig =synth_data.reshape(train_z.shape).astype(np.float32)
        x_next = synth_data_pre_vae_decodig

     
        syn_num, syn_cat, syn_target = split_num_cat_target(x_next, info, num_inverse, cat_inverse, args.device) 
    
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)
    
        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    
        syn_df.rename(columns = idx_name_mapping, inplace=True)
        syn_df.to_csv(save_path, index = False)
        
        end_time = time.time()
        #print('Time:', end_time - start_time)
        
        return syn_df
     
    elif  args.method in ['simple_KDE', 'TabKDE']:
     
        '''
            Generating samples    
        '''
     
        start_time = time.time()

        save_dir = args.save_path #os.path.join(parent_dir, "synthetic", dataname)
        #os.makedirs(save_dir, exist_ok=True)  # make directory if it doesn't exist
        
        save_path = os.path.join(save_dir, f"{args.method}.csv")

        
        #print(f'methood is {args.method} with latent_encoding {args.latent_encoding} and save_path is {save_path}')

        train_z, curr_dir, dataset_dir, ckpt_path, info = get_input_train(args)
     
        #------------------------------
        #print(f'The generation method is {args.method}')
        
        #cprint(f'synthetic data will be saved on {save_path}', bold = True)

        
        npy_z = train_z.reshape(train_z.shape[0], -1)
        #print(f'train_z.shape is {npy_z.shape}')
        df_train_copula = pd.DataFrame(npy_z)
        n, _ = df_train_copula.shape
        X = df_train_copula.sample(n=n, random_state=None, replace = True).values

     
        gmm_BIC = info['mix_gaussian']
        sample_z = perturb_vector_with_partial_resampling_with_DCP(X, 
                                                                  sigma = sigma, 
                                                                  num_samples = args.num_samples,
                                                                  model = gmm_BIC, 
                                                                  allowed_outside_boundary = True if args.method =='simple_KDE' else False)

        synth_data = convert_sample_z_to_df(sample_z, args)

        # Ensure the folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        #synth_data.to_csv(save_path, index=False)
        #cprint(f'synthetic data has been successfully  saved on {save_path}', bold = True, color = 'red')
        
        end_time = time.time()
        #print('Time:', end_time - start_time)
        return synth_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'