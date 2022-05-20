import pathlib

import matplotlib.pyplot as plt

from utils.svd import *

EXTRA_LARGE_FONTDICT = {"fontsize": 34}
LARGE_FONTDICT = {"fontsize": 30}


def plot_isometry_generic(Z1: torch.Tensor, Z2: torch.Tensor, title: str, file: pathlib.Path,
                          eps: float = 1e-8):  # (n, d),  (n, d)
    plt.rcParams['figure.figsize'] = 8, 8

    d1 = torch.clamp(torch.cdist(Z1, Z1), min=eps)  # (n, n)
    d2 = torch.cdist(Z2, Z2)  # (n, n)
    r = (d2 / d1).detach().numpy()  # (n, n)
    rs = [r[i][j] for i in range(r.shape[0]) for j in range(i)]
    plt.title(title, fontdict=EXTRA_LARGE_FONTDICT)
    plt.hist(rs, bins=10)
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def plot_isometry_X_fX(X: torch.Tensor, fX: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_isometry_generic(Z1=X, Z2=fX, title="$\\frac{\|f_{\star}(x^{i}) - f_{\star}(x^{j})\|}{\|x^{i} - x^{j}\|}$",
                          file=results_folder / "X_fX_isometry.jpg")


def plot_proj_residual_generic(Z1: torch.Tensor, Z2: torch.Tensor,
                               title: str, file: pathlib.Path):
    plt.rcParams['figure.figsize'] = 8, 8

    plt.suptitle(title, fontsize=EXTRA_LARGE_FONTDICT["fontsize"])
    P_Col_Z2 = projection_onto_col(Z2.T)  # (d, d)
    Z1_proj = Z1 @ P_Col_Z2  # (n, d)
    residuals = torch.linalg.norm(Z1 - Z1_proj, dim=1)  # (n, )
    plt.hist(residuals.detach().numpy())
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def plot_proj_residual_fX_fgfX(fX: torch.Tensor, fgfX: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_proj_residual_generic(Z1=fX, Z2=fgfX,
                               title="$\mathrm{Resid}(f_{\star}(X), \mathrm{Col}((f_{\star} \circ g_{\star} \circ f_{\star})(X)))$",
                               file=results_folder / "fX_fgfX_proj.jpg")
