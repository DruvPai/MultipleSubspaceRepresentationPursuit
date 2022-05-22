import pytorch_lightning as pl

from model.comparisons.operators import *


class UnsupervisedVAE(pl.LightningModule):
    def autoencode_data(self, X: torch.Tensor):
        raise NotImplementedError


class SupervisedVAE(pl.LightningModule):
    def autoencode_data(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError


class UnsupervisedVanillaVAE(UnsupervisedVAE):
    def __init__(self, d_x: int, d_z: int, d_latent: int, n_layers: int, lr: float):
        super(UnsupervisedVanillaVAE, self).__init__()
        self.d_x: int = d_x
        self.d_z: int = d_z
        self.encoder: torch.nn.Module = fcnn(d_x, d_latent, d_latent, n_layers - 1)
        self.fc_mu: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.fc_var: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.decoder: torch.nn.Module = fcnn(d_z, d_x, d_latent, n_layers)
        self.lr: float = lr
        self.training_loss = []
        self.name = f"VanillaVAE_dx{d_x}_dz{d_z}_dl{d_latent}_l{n_layers}_lr{lr}"

    def forward(self, x):
        x_ll = self.encoder(x)
        mu = self.fc_mu(x_ll)
        log_var = self.fc_var(x_ll)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        x_ll = self.encoder(x)
        mu = self.fc_mu(x_ll)
        log_var = self.fc_var(x_ll)
        p, q, z = self.sample(mu, log_var)
        x_hat = self.decoder(z)
        recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        loss = kl + recon_loss
        self.log_dict({"recon_loss": recon_loss, "kl": kl, "loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def autoencode_data(self, X: torch.Tensor):
        return self.forward(X)


class SupervisedCVAE(SupervisedVAE):
    def __init__(self, k: int, d_x: int, d_z: int, d_latent: int, n_layers: int, lr: float):
        super(SupervisedCVAE, self).__init__()
        self.k: int = k
        self.d_x: int = d_x
        self.d_z: int = d_z
        self.label_embed: torch.nn.Module = torch.nn.Linear(k, k)
        self.data_embed: torch.nn.Module = torch.nn.Linear(d_x, d_x)
        self.encoder: torch.nn.Module = fcnn(d_x + k, d_latent, d_latent, n_layers)
        self.fc_mu: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.fc_var: torch.nn.Module = torch.nn.Linear(d_latent, d_z)
        self.decoder: torch.nn.Module = fcnn(d_z, d_x, d_latent, n_layers)
        self.lr: float = lr
        self.training_loss = []
        self.name = f"CVAE_k{k}_dx{d_x}_dz{d_z}_dl{d_latent}_l{n_layers}_lr{lr}"

    def forward(self, x, y):
        x_ll = self.encoder(
            torch.cat((self.data_embed(x), self.label_embed(torch.nn.functional.one_hot(y).float())), dim=-1))
        mu = self.fc_mu(x_ll)
        log_var = self.fc_var(x_ll)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_ll = self.encoder(
            torch.cat((self.data_embed(x), self.label_embed(torch.nn.functional.one_hot(y).float())), dim=-1))
        mu = self.fc_mu(x_ll)
        log_var = self.fc_var(x_ll)
        p, q, z = self.sample(mu, log_var)
        x_hat = self.decoder(z)
        recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        loss = kl + recon_loss
        self.log_dict({"recon_loss": recon_loss, "kl": kl, "loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def autoencode_data(self, X: torch.Tensor, y: torch.Tensor):
        return self.forward(X, y)
