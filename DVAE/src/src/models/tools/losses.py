import torch
import torch.nn.functional as F
from .hessian_penalty import hessian_penalty
from .mmd import compute_mmd
import torch.nn as nn
loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()
softmax_cl = nn.Softmax(dim=2)

cosine_sim = nn.CosineSimilarity(dim=2, eps=1e-6)

def compute_rc_loss(model, batch):
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]

    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_contrastive_loss(model,batch):
    contrastive_x =softmax_cl( batch["contrastive_x"])
    contrastive_mesh =softmax_cl( batch["contrastive_mesh"][:60,:,:])

    cos = cosine_sim(contrastive_x, contrastive_mesh)
    cosine_loss = (1 - cos).mean()
    return cosine_loss*0.1

def compute_rcxyz_loss(model, batch):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]

    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss


def compute_vel_loss(model, batch):
    x = batch["x"]
    output = batch["output"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]
    
    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]
    
    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss


def compute_velxyz_loss(model, batch):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]
    
    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]
    
    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss


def compute_hp_loss(model, batch):
    loss = hessian_penalty(model.return_latent, batch, seed=torch.random.seed())
    return loss


def compute_kl_loss(model, batch):
    mu, logvar = batch["mu"], batch["logvar"]
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss


def compute_mmd_loss(model, batch):
    z = batch["z"]
    true_samples = torch.randn(z.shape, requires_grad=False, device=model.device)
    loss = compute_mmd(true_samples, z)
    return loss


_matching_ = {"rc": compute_rc_loss,"gcl":compute_contrastive_loss, "kl": compute_kl_loss, "hp": compute_hp_loss,
              "mmd": compute_mmd_loss, "rcxyz": compute_rcxyz_loss,
              "vel": compute_vel_loss, "velxyz": compute_velxyz_loss}


def get_loss_function(ltype):
    return _matching_[ltype]


def get_loss_names():
    return list(_matching_.keys())
