import typing

import pytorch_lightning as pl

from data.alterations import *


def _remove_spaces(x):
    return str(x).replace(" ", "")


class MultipleSubspacesDataModule(pl.LightningDataModule):
    def __init__(self, n: typing.List[int], k: int, d_x: int, d_S: typing.List[int], nu: float,
                 sigma_sq: float = 0.0, outlier_pct: float = 0.0, outlier_mag: float = 0.0,
                 label_corruption_pct: float = 0.0, batch_size: int = 50):

        super(MultipleSubspacesDataModule, self).__init__()

        self.n: typing.List[int] = n
        self.n_tot: int = sum(n)

        self.k: int = k

        self.d_x: int = d_x
        self.d_S: typing.List[int] = d_S
        self.d_S_tot: int = sum(d_S)

        self.nu: float = nu
        self.sigma_sq: float = sigma_sq
        self.batch_size: int = batch_size

        self.outlier_pct: float = outlier_pct
        self.outlier_mag: float = outlier_mag
        self.label_corruption_pct: float = label_corruption_pct

        self.name: str = f"multisub_n{_remove_spaces(n)}_k{k}_dx{d_x}_dS{_remove_spaces(d_S)}_nu{nu}_ss{sigma_sq}_op{outlier_pct}_om{outlier_mag}_lcp{label_corruption_pct}_bs{batch_size}"

    def setup(self, stage=None):
        self.X_train = torch.zeros(size=(self.n_tot, self.d_x))
        self.X_val = torch.zeros(size=(self.n_tot, self.d_x))
        self.X_test = torch.zeros(size=(self.n_tot, self.d_x))
        self.y_train = torch.zeros(self.n_tot, dtype=torch.int64)
        self.y_val = torch.zeros(self.n_tot, dtype=torch.int64)
        self.y_test = torch.zeros(self.n_tot, dtype=torch.int64)

        max_d_S = max(self.d_S)
        Q, R = torch.linalg.qr(torch.randn((self.d_x, self.d_x)))  # (d_x, d_x)
        U = Q[:, :max_d_S]  # (d_x, max_j d_S_j)

        Ujs = []
        for j in range(self.k):
            subset_columns = torch.multinomial(input=torch.ones(max_d_S), num_samples=self.d_S[j],
                                               replacement=False)  # (d_S_i, )
            Uj = U[:, subset_columns]  # (d_x, d_S_i)
            Ujp = (1 - self.nu) * Uj + self.nu * torch.randn_like(Uj)  # (d_x, d_S_i)
            Ujp = Ujp / torch.linalg.norm(Ujp, dim=0)  # (d_x, d_S_i)
            Ujs.append(Ujp)

        for (X, y) in ((self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)):
            num_processed = 0
            for j in range(self.k):
                X[num_processed:num_processed + self.n[j]] = (Ujs[j] @ torch.randn(size=(self.d_S[j], self.n[j]))).T
                y[num_processed:num_processed + self.n[j]] = j
                num_processed += self.n[j]
            add_noise(X, self.sigma_sq)
            add_outliers(X, self.outlier_pct, self.outlier_mag)
            add_labelcorruption(y, self.k, self.label_corruption_pct)

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
