import typing

import mcr2
import pytorch_lightning as pl

from model.operators import *


class SupervisedCTRLSG(pl.LightningModule):
    def __init__(self,
                 F: torch.nn.Module, G: torch.nn.Module,
                 E: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 C: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 inner_opt_steps: int = 1000):
        super(SupervisedCTRLSG, self).__init__()
        self.F: torch.nn.Module = F
        self.G: torch.nn.Module = G
        self.E: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = E
        self.C: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = C

        self.inner_opt_steps: int = inner_opt_steps

        self.training_E = []
        self.training_C = []

        self.automatic_optimization = False

    def f(self, X):
        return self.F(X)

    def gf(self, X):
        return self.G(self.F(X))

    def fgf(self, X):
        return self.F(self.G(self.F(X)))

    def u_enc(self, Z, Z_hat, Pi):
        return self.E(Z, Pi) - self.C(Z, Z_hat, Pi)

    def u_dec(self, Z, Z_hat, Pi):
        return self.C(Z, Z_hat, Pi)

    def training_step(self, batch, batch_idx):
        F_opt, G_opt = self.optimizers()

        X, y = batch
        Pi = mcr2.functional.y_to_pi(y)

        # Optimize
        loss_F = -self.u_enc(self.f(X), self.fgf(X), Pi)
        F_opt.zero_grad()
        self.manual_backward(loss_F)
        F_opt.step()

        self.F.project_parameters(X, Pi)

        for i in range(self.inner_opt_steps):
            loss_G = -self.u_dec(self.f(X), self.fgf(X), Pi)
            G_opt.zero_grad()
            self.manual_backward(loss_G)
            G_opt.step()

            self.G.project_parameters(X, Pi)

        # Log
        Z = self.f(X)
        Z_hat = self.fgf(X)
        Ef = self.E(Z, Pi)
        Cfg = self.C(Z, Z_hat, Pi)

        self.training_E.append(Ef.detach().numpy())
        self.training_C.append(Cfg.detach().numpy())
        self.log_dict({"E(f)": Ef, "C(f, g)": Cfg}, prog_bar=True)

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
            E=lambda Z, Pi: cr.DeltaR(Z, Pi),
            C=lambda Z1, Z2, Pi: -sum(
                cr.DeltaR_distance(Z1[Pi[:, j] == 1], Z2[Pi[:, j] == 1])
                for j in range(Pi.shape[1])
            ),
            inner_opt_steps=inner_opt_steps
        )
        self.training_DeltaR = self.training_E
        self.training_DeltaR_distance = [-self.training_C[i] for i in range(len(self.training_C))]

        self.name: str = f"CTRLMSP_dx{d_x}_dz{d_z}_es{eps_sq}_lrf{lr_f}_lrg{lr_g}_in{inner_opt_steps}"
