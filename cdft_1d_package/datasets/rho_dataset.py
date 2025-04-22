import os
import numpy as np
from torch.utils.data import Dataset
from cdft_1d_package.utils.data_split import load_train_val_split 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_data(files, path, cutoff, dz):

    rho_list = []
    Vext_list = []
    epsilon_list = []
    mu_list = []
    dF_drho_list = []

    for i in files:

        rho = np.load(f"{path}/rho_"+str(i)+".npy")
        V_ext = np.load(f"{path}/Vext_"+str(i)+".npy")
        epsilon = np.float32(0.5) # not used
        mu = np.load(f"{path}/mu_"+str(i)+".npy")
        
        dF_drho = np.zeros_like(rho, dtype='float32')
        dF_drho[rho>cutoff] = (mu - V_ext[rho>cutoff] - np.log(rho[rho>cutoff])) 

        if dz != 1/100:
            rho = convert_dz(rho, dz=dz, L=10)
            V_ext = convert_dz(V_ext, dz=dz, L=10)
            dF_drho = convert_dz(dF_drho, dz=dz, L=10)

        rho_list.append(rho)
        dF_drho_list.append(dF_drho)
        Vext_list.append(V_ext)
        epsilon_list.append(epsilon)
        mu_list.append(mu) 

    return rho_list, Vext_list, epsilon_list, mu_list, dF_drho_list

def convert_dz(data, dz=1/100, L=10):

    z_values = np.linspace(0, L, len(data))
    data_interp = interp1d(z_values, data, kind='linear', bounds_error=False, fill_value=0.0)
    recast_z_values = np.arange(0, L, dz)
    recast_data = data_interp(recast_z_values).astype(np.float32)
    return recast_data

def get_datasets(config, val_only=False, jax=False):

    train_files, val_files = load_train_val_split(config.paths.datasplit_path, id="f1")

    if not val_only:
        assert config.paths.train_path == config.paths.val_path, "train_path and val_path must be the same for F1"

    train_set = cDFTDataset(train_files, config.paths.val_path, cutoff=config.data.cutoff, dz=config.data.dz, jax=jax)
    val_set = cDFTDataset(val_files, config.paths.val_path, cutoff=config.data.cutoff, dz=config.data.dz, jax=jax)

    if val_only:
        return val_set
    else:
        return train_set, val_set

def get_complete_dataset(dataset_path):

    file_nrs = np.array([int(f[4:-4]) for f in os.listdir(dataset_path) if f.startswith("rho_")])
    complete_set = cDFTDataset(file_nrs, dataset_path)

    return complete_set

class cDFTDataset(Dataset):
    def __init__(self, set_idx, dataset_path, cutoff, dz, jax=False):
        self.densities, self.Vexts, self.epsilons, self.mus, self.dF_drhos = load_data(set_idx, path=dataset_path, cutoff=cutoff, dz=dz)
        self.channel_dim = -1 if jax else 0

    def __len__(self):
        return len(self.densities)
    
    def __getitem__(self, idx):

        density = self.densities[idx]
        dF_drho = self.dF_drhos[idx]

        if np.random.rand() < 0.5:
            density = np.flip(density).copy()
            dF_drho = np.flip(dF_drho).copy()
        translation = np.random.randint(0,len(density))
        density = np.roll(density, translation)
        dF_drho = np.roll(dF_drho, translation)

        density = np.expand_dims(density, axis=self.channel_dim) # adding an empty channels dimension 
        dF_drho = np.expand_dims(dF_drho, axis=self.channel_dim)

        return density, dF_drho