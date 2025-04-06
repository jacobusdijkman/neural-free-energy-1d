import os
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_data(files, path):

    rho_list = []
    mu_list = []
    dF_drho_list = []
    c2_list = []

    for i in files:

        # bulk density 
        rho_b = np.load(f"{path}/rhob_"+str(i)+".npy")
        c2 = np.load(f"{path}/c2_"+str(i)+".npy")
        mu = np.load(f"{path}/mu_"+str(i)+".npy")
        dF_drho = np.load(f"{path}/dF_drho_"+str(i)+".npy")
        z_values = np.load(f"{path}/r_values_"+str(i)+".npy")
        
        recast_c2, recast_z_values = convert_dr(c2, z_values)
        c2 = expand_c2(recast_c2)

        rho_list.append(rho_b)
        dF_drho_list.append(dF_drho)
        mu_list.append(mu) 
        c2_list.append(c2)

    return rho_list, mu_list, dF_drho_list, c2_list

def expand_c2(c_z):

    c_z_mirrored = np.flip(c_z, axis=0)[:-1]
    c_z = np.concatenate((c_z, c_z[-1:], c_z_mirrored), axis=0)

    c2 = np.zeros((len(c_z), len(c_z)), dtype=np.float32)
    for i in range(len(c_z)):
        c2[i,:] = np.roll(c_z, i, axis=0)

    return c2

def convert_dr(c2, r_values, dz=1/100, R=5):

    c2_interp = interp1d(r_values, c2, kind='linear', bounds_error=False, fill_value=0.0)
    recast_r_values = np.arange(0, R, dz)
    recast_c2 = c2_interp(recast_r_values).astype(np.float32)
    return recast_c2, recast_r_values

def get_complete_dataset(config, jax=False):

    file_nrs = np.array([int(f.split('_')[-1][:-4]) for f in os.listdir(config.paths.train_path) if f.startswith("rhob_")])
    complete_set = cDFTDataset(file_nrs, config.paths.train_path, config.data.input_size, n_samples=config.trainer.d2F_samples, jax=jax)

    return complete_set

class cDFTDataset(Dataset):
    def __init__(self, set_idx, dataset_path, input_size, n_samples, jax=False):
        self.densities, self.mus, self.dF_drhos, self.c2s = load_data(set_idx, path=dataset_path)
        self.input_size = input_size
        self.n_samples = n_samples
        self.channel_dim = -1 if jax else 0

    def __len__(self):
        return len(self.densities)
    
    def __getitem__(self, idx):
        rand_idx = np.random.randint(0, self.input_size[0], size=self.n_samples)
        density = np.expand_dims(np.ones(self.input_size,dtype=np.float32)*self.densities[idx], axis=self.channel_dim)
        dF = np.expand_dims(np.ones(self.input_size,dtype=np.float32)*self.dF_drhos[idx], axis=self.channel_dim)
        d2F = np.expand_dims(- self.c2s[idx][rand_idx, :], axis=self.channel_dim)
        return density, dF, d2F, rand_idx
    
    def get_val_item(self, idx):
        center_idx = np.expand_dims(np.array(int(self.input_size[0]/2)), axis=(self.channel_dim+1, self.channel_dim))
        density = np.expand_dims(np.ones(self.input_size,dtype=np.float32)*self.densities[idx], axis=(self.channel_dim+1,self.channel_dim))
        dF = np.expand_dims(np.ones(self.input_size,dtype=np.float32)*self.dF_drhos[idx], axis=(self.channel_dim+1,self.channel_dim))
        d2F = np.expand_dims(- self.c2s[idx][center_idx, :], axis=self.channel_dim)
        return [density, dF, d2F, center_idx] 

