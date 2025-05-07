import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Dict
from cdft_1d_package.utils.setup import debugger_is_active
from cdft_1d_package.utils.plotting_utils import log_dF

@dataclass
class DataConfig:
    train_dataset: Optional[str] = None
    val_dataset: Optional[str] = None
    input_size: List[int] = field(default_factory=lambda: [1000])
    cutoff: float = 1e-3
    dz: float = 1 / 100  # the current cnn with default c_hidden assumes L=10 and dz=1/100. Changing this means changing the cnn hparams. 

@dataclass
class IDConfig:
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    note: str = ""


@dataclass
class TrainerConfig:
    trainer_seed: int = 44
    data_seed: int = 44
    batch_size: int = 32
    accumulate_batch_size: int = 32
    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3
    check_val_every_n_epoch: int = 10
    dF_alpha: float = 1
    d2F_alpha: float = 1
    d2F_samples: int = 10
    devices: int = 1
    log_every_n_steps: int = 10
    max_epochs: int = 100000
    num_train_workers : int = 8
    num_val_workers : int = 8
    matmul_precision: str = "highest"
    log_functions: Dict[str, Callable] = field(default_factory=lambda: {"dF": log_dF})
    viz: List[str] = field(default_factory=lambda: [])  # "gradients", "activations", "weights"
    debugger_active: bool = field(default_factory=debugger_is_active)

    def to_dict(self):
        return asdict(self)

@dataclass
class ModelConfig:
    c_hidden: List[int] = field(default_factory=lambda: [32,32,32,32,64,64,64,64])
    kernel_size: int = 3
    dilation: int = 3
    stride: int = 1
    padding: int = 3

    def to_dict(self):
        return asdict(self)

@dataclass
class PathsConfig:
    checkpoint_path: str = field(default="")
    dataset_path: str = field(default="")
    datasplit_path: str = field(default="")
    train_path: str = "set in set_paths()"
    val_path: str = "set in set_paths()"
    train_script_path: str = field(default_factory=lambda: os.getcwd())

    def __post_init__(self):
        """Initialize paths based on project root."""
        project_root = self.find_project_root()
        
        if not self.checkpoint_path:
            self.checkpoint_path = os.path.join(project_root, "logs/")
        
        if not self.dataset_path:
            self.dataset_path = os.path.join(project_root, "datasets/")
        
        if not self.datasplit_path:
            self.datasplit_path = os.path.join(project_root, "data_split/")

    def find_project_root(self):
        """
        Find the project root directory (neural-free-energy-1d).
        """
        current_dir = os.getcwd()
        
        # Walk up the directory tree to find the project root
        path = current_dir
        while path != '/' and path != '':  # Added empty string check for Windows
            # Check if this is the neural-free-energy-1d directory
            if os.path.basename(path) == 'neural-free-energy-1d':
                return path
                
            # Also check if there's a neural-free-energy-1d directory in the current path
            nfe_dir = os.path.join(path, "neural-free-energy-1d")
            if os.path.exists(nfe_dir):
                return nfe_dir
            
            # Move up one directory
            path = os.path.dirname(path)
        
        # If we can't find the project root, use the current directory
        return current_dir

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    id: IDConfig = field(default_factory=IDConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model_hparams: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    def to_dict(self):
        return {
            k: v.to_dict() if hasattr(v, 'to_dict') else v 
            for k, v in self.__dict__.items()
        }
    