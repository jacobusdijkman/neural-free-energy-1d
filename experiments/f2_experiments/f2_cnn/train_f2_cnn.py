import torch
from cdft_1d_package.utils.config import Config, DataConfig, IDConfig, TrainerConfig, ModelConfig, PathsConfig
from cdft_1d_package.utils.setup import setup, get_dataloaders
from cdft_1d_package.utils.base_module import BaseModule

# custom imports per experiment
from cdft_1d_package.models.cnn import CNN
from cdft_1d_package.datasets.c2_dataset import get_complete_dataset as get_train_dataset
from cdft_1d_package.datasets.rho_dataset import get_datasets as get_val_dataset
from cdft_1d_package.utils.plotting_utils import log_dF, log_d2F

class cDFT_CNN(BaseModule):

    def __init__(self, model : torch.nn.Module, config : Config):

        super().__init__(model, config)

    def d2F_loss(self, dF_pred, d2F_true, rho_true, rand_idx):
        d2F_pred_list = []
        for i in range(rand_idx.shape[1]):
            rand_idx_slice = rand_idx[:,i]
            batch_size = dF_pred.shape[0]
            batch_idx = torch.arange(batch_size, device=dF_pred.device)
            grad_elem = dF_pred[batch_idx, :, rand_idx_slice[batch_idx]][..., None]
            d2F_pred = torch.autograd.grad(grad_elem, rho_true, grad_outputs=torch.ones_like(grad_elem), create_graph=True)[0] / self.config.data.dz
            d2F_pred_list.append(d2F_pred)
        d2F_pred = torch.cat(d2F_pred_list, dim=1)
        d2F_loss = self.config.trainer.d2F_alpha*self.mse_loss(d2F_pred, d2F_true.squeeze(1))
        return d2F_loss, d2F_pred 
    
    def training_step(self, batch):
        rho_true, dF_true, d2F_true, rand_idx = batch
        rho_true.requires_grad = True
        with torch.set_grad_enabled(True):
            F_pred = self.forward(rho_true)
            dF_loss, dF_pred = self.dF_loss(F_pred, dF_true, rho_true)
            d2F_loss, _ = self.d2F_loss(dF_pred, d2F_true, rho_true, rand_idx)
        loss = dF_loss + d2F_loss
        self.log('train_loss', loss)
        self.log('dF_loss', dF_loss)
        self.log('d2F_loss', d2F_loss)
        return loss

if __name__ == "__main__":

    """Training a neural free energy using pair-correlation matching."""
    
    config = Config(id=IDConfig(project_name="1D_Vext_Experiments", 
                                run_name="f2_cnn",
                                note="<note>"), 
                    trainer=TrainerConfig(
                        check_val_every_n_epoch=10,
                        log_functions={"dF" : log_dF,
                                       "d2F" : log_d2F},
                        seed=12
                    ), 
                    data=DataConfig(
                        train_dataset="c2_dataset",
                        val_dataset="rho_dataset",
                        dz=1/100
                    ),
    )

    trainer = setup(config)
    train_set = get_train_dataset(config)
    val_set = get_val_dataset(config, val_only=True)
    train_loader, val_loader = get_dataloaders(train_set, val_set, config)
    model = CNN(**config.model_hparams.to_dict())
    module = cDFT_CNN(model, config)
    trainer.fit(module, train_loader, val_loader)

