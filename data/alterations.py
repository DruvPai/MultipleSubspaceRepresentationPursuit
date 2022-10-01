import torch


def add_noise(X: torch.Tensor, nu: float):
    n, d_x = X.shape
    X[:] += ((nu / d_x) ** 0.5) * torch.randn(X.shape)


def add_outliers(X: torch.Tensor, outlier_pct: float, outlier_mag: float):
    n, d_x = X.shape
    n_outlier = min(n, int(n * outlier_pct))
    if n_outlier > 0:
        idx_outliers = torch.multinomial(
            input=torch.ones(size=(X.shape[0],)),
            num_samples=n_outlier,
            replacement=False
        )
        X[idx_outliers] = ((outlier_mag / d_x) ** 0.5) * torch.randn(n_outlier, d_x)


def add_labelcorruption(y: torch.Tensor, k: int, lcr: float):
    n = y.shape[0]
    n_lc = min(n, int(lcr * y.shape[0]))
    if n_lc > 0:
        idx_lc = torch.multinomial(
            input=torch.ones(size=(n,)),
            num_samples=n_lc,
            replacement=False
        )
        y[idx_lc] = torch.randint(low=0, high=k, size=(n_lc,))
