import typing

import mcr2
import pytorch_lightning as pl

from model.operators import *


class SupervisedCTRLSG(pl.LightningModule):
    def __init__(self,
                 F: torch.nn.Module, G: torch.nn.Module,
                 Q: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 C: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 inner_opt_steps: int = 1000):
        super(SupervisedCTRLSG, self).__init__()
        self.F: torch.nn.Module = F
        self.G: torch.nn.Module = G
        self.Q: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = Q
        self.C: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = C

        self.inner_opt_steps: int = inner_opt_steps

        self.training_Q = []
        self.training_C = []

        self.automatic_optimization = False

    def f(self, X):
        return self.F(X)

    def gf(self, X):
        return self.G(self.f(X))

    def fgf(self, X):
        return self.F(self.gf(X))

    def u_enc(self, Z, Z_hat, Pi):
        return self.Q(Z, Pi) - self.C(Z, Z_hat, Pi)

    def u_dec(self, Z, Z_hat, Pi):
        return self.C(Z, Z_hat, Pi)

    def training_step(self, batch, batch_idx):
        F_opt, G_opt = self.optimizers()

        with torch.no_grad():
            X, y = batch
            Pi = mcr2.functional.y_to_pi(y)

        # Optimize
        Z = self.f(X)
        Z_hat = self.fgf(X)
        loss_F = - self.u_enc(Z, Z_hat, Pi)
        F_opt.zero_grad()
        self.manual_backward(loss_F)
        F_opt.step()

        with torch.no_grad():
            self.F.project_parameters(X, Pi)

        for i in range(self.inner_opt_steps):
            Z = self.f(X)
            Z_hat = self.fgf(X)
            loss_G = - self.u_dec(Z, Z_hat, Pi)
            G_opt.zero_grad()
            self.manual_backward(loss_G)
            G_opt.step()

            with torch.no_grad():
                self.G.project_parameters(X, Pi)

        # Log
        with torch.no_grad():
            Z = self.f(X)
            Z_hat = self.fgf(X)
            Qf = self.Q(Z, Pi)
            Cfg = self.C(Z, Z_hat, Pi)

        self.training_Q.append(Qf.detach().numpy())
        self.training_C.append(Cfg.detach().numpy())
        self.log_dict({"Q(f)": Qf, "C(f, g)": Cfg}, prog_bar=True)

    def configure_optimizers(self):
        F_opt = self.F.optimizer
        G_opt = self.G.optimizer
        return F_opt, G_opt


class CTRLMSP(SupervisedCTRLSG):
    def __init__(self, d_x: int, d_z: int, eps_sq: float = 1.0,
                 lr_f: float = 1e-2, lr_g: float = 1e-3, inner_opt_steps: int = 1000):
        cr = mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)
        super(CTRLMSP, self).__init__(
            F=LinearSampleNormConstrainedEncoder(d_x, d_z, lr_f),
            G=LinearUnconstrainedDecoder(d_x, d_z, lr_g),
            Q=lambda Z, Pi: cr.DeltaR(Z, Pi),
            C=lambda Z1, Z2, Pi: -sum(
                cr.DeltaR_distance(Z1[Pi[:, j] == 1], Z2[Pi[:, j] == 1])
                for j in range(Pi.shape[1])
            ),
            inner_opt_steps=inner_opt_steps
        )
        self.training_DeltaR = self.training_Q
        self.training_DeltaR_distance = [-self.training_C[i] for i in range(len(self.training_C))]

        self.name: str = f"CTRLMSP_dx{d_x}_dz{d_z}_es{eps_sq}_lrf{lr_f}_lrg{lr_g}_in{inner_opt_steps}"


class CTRLMSPFCNN(SupervisedCTRLSG):
    def __init__(self, d_x: int, d_z: int, d_latent: int, n_layers: int,
                 eps_sq: float = 1.0, lr_f: float = 1e-1, lr_g: float = 1e-3,
                 inner_opt_steps: int = 100):
        cr = mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)
        super(CTRLMSPFCNN, self).__init__(
            F=FCNNSampleNormConstrainedEncoder(d_x, d_z, d_latent, n_layers, lr_f),
            G=FCNNDecoder(d_x, d_z, d_latent, n_layers, lr_g),
            Q=lambda Z, Pi: cr.DeltaR(Z, Pi),
            C=lambda Z1, Z2, Pi: -sum(
                cr.DeltaR_distance(Z1[Pi[:, j] == 1], Z2[Pi[:, j] == 1])
                for j in range(Pi.shape[1])
            ),
            inner_opt_steps=inner_opt_steps
        )
        self.training_DeltaR = self.training_Q
        self.training_DeltaR_distance = [-self.training_C[i] for i in range(len(self.training_C))]

        self.name: str = f"CTRLMSPFCNN_dx{d_x}_dz{d_z}_dl{d_latent}_nl{n_layers}_es{eps_sq}_lrf{lr_f}_lrg{lr_g}_in{inner_opt_steps}"
