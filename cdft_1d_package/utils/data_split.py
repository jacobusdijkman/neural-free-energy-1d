import os
import numpy as np
import torch
import pytorch_lightning as pl

def get_train_val_split(dataset_path):
    file_nrs = np.array([int(f[4:-4]) for f in os.listdir(dataset_path) if f.startswith("rho_")])
    n_samples = len(file_nrs)
    n_train_samples = int(4*n_samples/5)
    n_val_samples = n_samples - n_train_samples
    train_files, val_files = torch.utils.data.random_split(file_nrs, [n_train_samples, n_val_samples])
    return file_nrs[train_files.indices], file_nrs[val_files.indices]

def save_train_val_split(train_files, val_files, save_datasplit_path, id):
    if not os.path.exists(save_datasplit_path):
        os.makedirs(save_datasplit_path, exist_ok=True)
    
    with open(f"{save_datasplit_path}/{id}_train_files.txt", 'w') as f:
        f.write('\n'.join(map(str, train_files)))
    
    with open(f"{save_datasplit_path}/{id}_val_files.txt", 'w') as f:
        f.write('\n'.join(map(str, val_files)))

def load_train_val_split(datasplit_path, id):
    with open(f"{datasplit_path}/{id}_train_files.txt", 'r') as f:
        train_files = np.array([int(x) for x in f.read().splitlines()])
    
    with open(f"{datasplit_path}/{id}_val_files.txt", 'r') as f:
        val_files = np.array([int(x) for x in f.read().splitlines()])
    
    return train_files, val_files

if __name__ == "__main__":

    # this is to have a consistent train/val split across different runs
    # this is really only necessary for the rho dataset, 
    # since the c2 dataset is completely used during training, 
    # because the rho dataset is used for validation.

    SEED = 44
    DATASET_PATH = "/path/to/dataset/"
    SAVE_DATASPLIT_PATH = "/path/to/package/cdft_1d_package/data_split"
    ID = "f1"
    
    pl.seed_everything(44)
    train_files, val_files = get_train_val_split(DATASET_PATH)
    save_train_val_split(train_files, val_files, SAVE_DATASPLIT_PATH, ID)