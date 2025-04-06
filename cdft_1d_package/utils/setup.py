import sys
import os
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.utils.data as data
import pytorch_lightning as pl

def debugger_is_active() -> bool:
    has_breakpoint = sys.breakpointhook.__module__ != "sys"
    has_trace_func = sys.gettrace() is not None
    debugger_is_active = (has_breakpoint or has_trace_func)
    return debugger_is_active

def set_paths(config):

    config.paths.checkpoint_path = config.paths.local_checkpoint_path
    config.paths.train_path = config.paths.local_dataset_path + config.data.train_dataset
    config.paths.val_path = config.paths.local_dataset_path + config.data.val_dataset
    config.paths.datasplit_path = config.paths.local_datasplit_path
    os.makedirs(config.paths.checkpoint_path, exist_ok=True)

def set_wandb(config):

    offline_run = debugger_is_active()

    wandb_logger = WandbLogger(project=config.id.project_name, 
                            name=config.id.run_name, 
                            save_dir=config.paths.checkpoint_path, 
                            offline=offline_run,
                            config=config.to_dict(),
                            experiment=wandb.run)  
    config.trainer.run_id = wandb_logger.experiment.id

    return wandb_logger

def set_trainer(wandb_logger, config):

    trainer = pl.Trainer(default_root_dir=os.path.join(config.paths.checkpoint_path, config.id.project_name),                                  
                        accelerator="auto",                                                                           
                        devices=1,
                        check_val_every_n_epoch = config.trainer.check_val_every_n_epoch if not debugger_is_active() else 1,
                        log_every_n_steps=config.trainer.log_every_n_steps,                                                                         
                        max_epochs=config.trainer.max_epochs,
                        enable_progress_bar=True,
                        logger=wandb_logger,  
                        accumulate_grad_batches=int(config.trainer.accumulate_batch_size/config.trainer.batch_size), 
                        callbacks=[
                            ModelCheckpoint(save_weights_only=False,mode="min",monitor="val_loss",save_last=True),           
                            LearningRateMonitor("epoch")]
                        )    
                             
    
    return trainer

def get_dataloaders(train_set, val_set, config, collate_fn=None):

    data_generator = torch.Generator().manual_seed(config.trainer.data_seed) # the random number generator of the data loader

    train_loader = data.DataLoader(train_set, 
                                   batch_size=config.trainer.batch_size, 
                                   shuffle=True, 
                                   drop_last=True, 
                                   pin_memory=False, 
                                   num_workers = 0, 
                                   generator=data_generator,
                                   collate_fn=collate_fn)
    val_loader = data.DataLoader(val_set, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 drop_last=False, 
                                 num_workers = 0,
                                 collate_fn=collate_fn,) 

    return train_loader, val_loader

def setup(config):

    if debugger_is_active():
        print("-- Debugger is active ---")

    torch.set_float32_matmul_precision(config.trainer.matmul_precision)
    pl.seed_everything(config.trainer.trainer_seed) 

    set_paths(config)
    wandb_logger = set_wandb(config)
    trainer = set_trainer(wandb_logger, config)

    return trainer
