import torch


def fcnn(d_in: int, d_out: int, d_latent: int, n_layers: int):
    layers = [torch.nn.Linear(d_in, d_latent), torch.nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.extend([torch.nn.Linear(d_latent, d_latent), torch.nn.ReLU()])
    layers.append(torch.nn.Linear(d_latent, d_out))
    return torch.nn.Sequential(*layers)
