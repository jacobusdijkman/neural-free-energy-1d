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
    dz: float = 1 / 100

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
    matmul_precision: str = "medium"
    log_functions: Dict[str, Callable] = field(default_factory=lambda: {"dF": log_dF})
    viz: List[str] = field(default_factory=lambda: [])  # "gradients", "activations", "weights"
    debugger_active: bool = field(default_factory=debugger_is_active)

    def to_dict(self):
        return asdict(self)

@dataclass
class ModelConfig:
    c_hidden: List[int] = field(default_factory=lambda: [32,32,32,32,64,64,64,64])
    downsampling_kernel_size: int = 3
    kernel_size: int = 3
    dilation: int = 3
    stride: int = 1
    padding: int = 3

    def to_dict(self):
        return asdict(self)

@dataclass
class PathsConfig:
    local_checkpoint_path: str = (
        "path/to/where/you/want/the/models/to/be/saved/"
    )
    local_dataset_path: str = (
        "/path/to/dataset/"
    )
    local_datasplit_path: str = (
        "/path/to/package/cdft_1d_package/data_split/"
    )
    train_path: str = "set in set_paths()"
    val_path: str = "set in set_paths()"
    train_script_path: str = field(default_factory=lambda: os.getcwd())

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
    