import torch
from cdft_1d_package.utils.config import *
from cdft_1d_package.utils.setup import setup, get_dataloaders
from cdft_1d_package.utils.base_module import BaseModule

# custom imports per experiment
from cdft_1d_package.models.cnn import CNN
from cdft_1d_package.datasets.rho_dataset import get_datasets 

class cDFT_CNN(BaseModule):

    def __init__(self, model : torch.nn.Module, config : Config):

        super().__init__(model, config)

if __name__ == "__main__":

    """Training a neural free energy using the inhomogeneous first functional derivative."""
    
    config = Config(id=IDConfig(project_name="neural_free_energy_experiments",
                                run_name="f1_cnn",
                                note="<note>"),
                    trainer=TrainerConfig(
                        check_val_every_n_epoch=10,
                        log_functions={"dF" : log_dF},
                    ), 
                    data=DataConfig(
                        train_dataset="rho_dataset",
                        val_dataset="rho_dataset",
                        dz=1/100,
                    )
    )

    trainer = setup(config)
    train_set, val_set = get_datasets(config)
    train_loader, val_loader = get_dataloaders(train_set, val_set, config)
    model = CNN(**config.model_hparams.to_dict())
    module = cDFT_CNN(model, config)
    trainer.fit(module, train_loader, val_loader)