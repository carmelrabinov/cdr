import os
import torch
import yaml
from easydict import EasyDict as edict
from torch.optim import Optimizer

from model.cpc import ControlCPC


def run_single(model: torch.nn.Module, *args) -> torch.Tensor:
    """ Runs a single element (no batch dimension) through a PyTorch model """
    return model(*[a.unsqueeze(0) for a in args]).squeeze(0)


def load_config(path: str) -> dict:
    """ load a config file """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return edict(config)


def dump_config(output_directory: str, config: dict) -> None:
    """ dump a config file """
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_from_checkpoint(path: str, config: dict, device: str):
    # load checkpoint
    print(f"Loading model from {path}")
    if device == torch.device('cpu'):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(path)

    # init and load model
    model = ControlCPC(config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.forward_model.load_state_dict(checkpoint['forward_model'])
    model.to(device)
    print("Done loading model!")

    # init and load optimizer
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']

    return model, optimizer, epoch + 1


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: Optimizer, epoch: int, val_loss: float):
    checkpoint = {
        'encoder': model.encoder.state_dict(),
        'forward_model': model.forward_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}"))
    print(f'Saved models with loss {val_loss} to {path}')


def freeze_weights(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


