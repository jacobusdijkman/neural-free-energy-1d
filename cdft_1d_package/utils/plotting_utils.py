import matplotlib.pyplot as plt
import torch
import wandb

def log_dF(module) -> None:

    module.model.eval()
    rho_true, dF_true = module.viz_dF_sample
    rho_true.requires_grad = True

    with torch.set_grad_enabled(True):
        F_pred = module.forward(rho_true)
        dF_pred = torch.autograd.grad(F_pred, rho_true, grad_outputs=torch.ones_like(F_pred), create_graph=True)[0] / module.config.data.dz

    figure, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.plot(dF_pred[0,0,:].detach().cpu(), label='dF_pred')
    ax1.plot(dF_true[0,0,:].detach().cpu(), label='dF_true')
    ax1.legend()
    ax2.plot(rho_true[0,0,:].detach().cpu())
    ax2.set_title('rho_mc')
    figure.tight_layout()
    figure.suptitle(f'dF experiment {module.logger.version} (epoch {module.current_epoch})')

    wandb.log({'dF plot': wandb.Image(figure)})
    plt.close('all')

def log_d2F(module) -> None:

    module.model.zero_grad()
    module.model.eval()
    rho_true, dF_true, d2F_true, _ = module.viz_d2F_sample
    rho_true.requires_grad = True

    middle_idx = int(module.config.data.input_size[0] // 2)

    with torch.set_grad_enabled(True):
        F_pred = module.forward(rho_true)
        dF_pred = torch.autograd.grad(F_pred, rho_true, grad_outputs=torch.ones_like(F_pred), create_graph=True)[0] / module.config.data.dz
        grad_elem = dF_pred[:,:,middle_idx]
        d2F_pred = torch.autograd.grad(grad_elem, rho_true, grad_outputs=torch.ones_like(grad_elem), create_graph=True)[0] / module.config.data.dz

    # Create figure with subplots
    figure, axes = plt.subplots(3, 1, figsize=(10, 12))

    figure.suptitle(f'd2F experiment {module.logger.version} (epoch {module.current_epoch})')

    # First subplot: dF comparison
    axes[0].plot(dF_true[0,0,:], label="dF_true")
    axes[0].plot(dF_pred[0,0,:].detach().cpu(), label="dF_pred")
    axes[0].set_title("First Derivative Comparison")
    axes[0].legend()
    
    # Second subplot: d2F comparison
    axes[1].plot(d2F_true[0,0,0,:].detach().cpu(), label="d2F_true")
    axes[1].plot(d2F_pred[0,0,:].detach().cpu(), label="d2F_pred")
    axes[1].set_title("Second Derivative Comparison")
    axes[1].legend()
    
    # Third subplot: Density profile
    axes[2].plot(rho_true[0,0,:].detach().cpu(), label="rho_true")
    axes[2].set_title("Density Profile")
    axes[2].legend()

    plt.tight_layout()
    wandb.log({'d2F image': wandb.Image(figure)})
    plt.close('all')