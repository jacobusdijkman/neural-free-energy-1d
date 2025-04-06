import pytorch_lightning as pl
import torch
import torch.optim as optim 
from torch.nn import MSELoss
from cdft_1d_package.utils.config import Config

class BaseModule(pl.LightningModule):

    def __init__(self, model : torch.nn.Module, config : Config):
        super().__init__()
        self.model = model
        self.config = config
        self.example_input_array = torch.zeros((config.trainer.batch_size,1,*config.data.input_size), dtype=torch.float32)
        self.mse_loss = MSELoss()
        self.train_start = False 

    def forward(self, x):
        z = self.model(x)
        return z
    
    def log_samples(self):  
        for log_fn in self.config.trainer.log_functions.values():
            log_fn(self)
    
    def on_train_start(self):
        if 'dF' in self.config.trainer.log_functions:
            self.init_val_dF_sample()
        if 'd2F' in self.config.trainer.log_functions:
            self.init_val_d2F_sample()
        
        self.train_start = True

    def init_val_dF_sample(self):
        val_dataloader = self.trainer.val_dataloaders
        self.viz_dF_sample = next(iter(val_dataloader))
        self.viz_dF_sample[0] = self.viz_dF_sample[0].to(self.device)
        return True

    def init_val_d2F_sample(self):
        self.viz_d2F_sample = self.trainer.train_dataloader.dataset.get_val_item(0)
        self.viz_d2F_sample[0] = torch.tensor(self.viz_d2F_sample[0]).to(self.device)
        self.viz_d2F_sample[1] = torch.tensor(self.viz_d2F_sample[1])
        self.viz_d2F_sample[2] = torch.tensor(self.viz_d2F_sample[2]).to(self.device)
        return True

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.trainer.learning_rate)
        return optimizer
    
    def dF_loss(self, F_pred, dF_true, rho_true, val=False):
        "default dF loss"
        alpha = self.config.trainer.dF_alpha if not val else 1
        dF_pred = torch.autograd.grad(F_pred, rho_true, torch.ones_like(F_pred) if F_pred.numel() > 1 else None, create_graph=True)[0] /  self.config.data.dz
        dF_loss = alpha*self.mse_loss(dF_pred[rho_true>self.config.data.cutoff],dF_true[rho_true>self.config.data.cutoff])
        return dF_loss, dF_pred

    def training_step(self, batch, batch_idx):  
        rho_true, dF_true = batch
        rho_true.requires_grad = True
        with torch.set_grad_enabled(True):
            F_pred = self.forward(rho_true)
            dF_loss, _ = self.dF_loss(F_pred, dF_true, rho_true)
        loss = dF_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        "default validation step"
        rho_true, dF_true = batch
        rho_true.requires_grad = True
        with torch.set_grad_enabled(True):
            F_pred = self.forward(rho_true)
            dF_loss, _ = self.dF_loss(F_pred, dF_true, rho_true, val=True)
        loss = dF_loss
        self.log('val_loss', loss)
        if batch_idx == 0 and self.train_start:
            self.log_samples()