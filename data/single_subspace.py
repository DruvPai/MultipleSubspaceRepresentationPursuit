import pytorch_lightning as pl

from data.alterations import *


class SingleSubspaceDataModule(pl.LightningDataModule):
    def __init__(self, n: int, d_x: int, d_S: int, nu: float = 0.0, outlier_pct: float = 0.0,
                 outlier_mag: float = 0.0, batch_size: int = 50):
        super(SingleSubspaceDataModule, self).__init__()

        self.d_x: int = d_x
        self.d_S: int = d_S
        self.n: int = n

        self.nu: float = nu
        self.batch_size: int = batch_size

        self.outlier_pct: float = outlier_pct
        self.outlier_mag: float = outlier_mag

        self.name: str = f"singlesub_n{n}_dx{d_x}_dS{d_S}_nu{nu}_op{outlier_pct}_om{outlier_mag}_bs{batch_size}"

    def setup(self, stage=None):
        self.X_train: torch.Tensor = torch.zeros(size=(self.n, self.d_x))
        self.X_val: torch.Tensor = torch.zeros(size=(self.n, self.d_x))
        self.X_test: torch.Tensor = torch.zeros(size=(self.n, self.d_x))

        Q, R = torch.linalg.qr(torch.randn((self.d_x, self.d_x)))  # (d_x, d_x), (d_x, d_x)
        U: torch.Tensor = Q[:, :self.d_S]  # (d_x, d_M)

        for X in (self.X_train, self.X_val, self.X_test):
            X[:] = (U @ torch.randn(self.d_S, X.shape[0])).T
            add_noise(X, self.nu)
            add_outliers(X, self.outlier_pct, self.outlier_mag)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.X_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.X_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.X_test, batch_size=self.batch_size)
