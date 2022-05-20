import pathlib
import typing

import matplotlib.pyplot as plt

from utils.svd import *

EXTRA_LARGE_FONTDICT = {"fontsize": 34}
LARGE_FONTDICT = {"fontsize": 30}


def plot_cosine_similarity_generic(Z: torch.Tensor, title: str, file: pathlib.Path, eps: float = 1e-8):  # (n, d)
    plt.rcParams['figure.figsize'] = 8, 8

    z_norms = torch.clamp(torch.linalg.norm(Z, dim=1)[:, None], min=eps)  # (n, 1)
    Z_normalized = Z / z_norms  # (n, d)
    cos_sim = Z_normalized @ Z_normalized.T  # (n, n)
    abs_cos_sim = torch.abs(cos_sim)  # (n, n)

    plt.title(title, fontdict=EXTRA_LARGE_FONTDICT)
    plt.imshow(abs_cos_sim.detach().numpy(), cmap="Blues")
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def plot_cosine_similarity_X(X: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_cosine_similarity_generic(
        Z=X,
        title="$\cos(\\angle(x^i, x^j))$",
        file=results_folder / "X_heatmap.jpg"
    )


def plot_cosine_similarity_fX(fX: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_cosine_similarity_generic(
        Z=fX,
        title="$\cos(\\angle(f_{\star}(x^i), f_{\star}(x^j)))$",
        file=results_folder / "fX_heatmap.jpg"
    )


def plot_cosine_similarity_gfX(gfX: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_cosine_similarity_generic(
        Z=gfX,
        title="$\cos(\\angle((g_{\star} \circ f_{\star})(x^i), (g_{\star} \circ f_{\star})(x^j))$",
        file=results_folder / "gfX_heatmap.jpg"
    )


def plot_cosine_similarity_fgfX(fgfX: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_cosine_similarity_generic(
        Z=fgfX,
        title="$\cos(\\angle((f_{\star} \circ g_{\star} \circ f_{\star})(x^i), (f_{\star} \circ g_{\star} \circ f_{\star})(x^j))$",
        file=results_folder / "fgfX_heatmap.jpg"
    )


def plot_spectra_generic(Z: torch.Tensor, title: str, file: pathlib.Path):
    plt.rcParams['figure.figsize'] = 8, 8

    spec = spectrum(Z)
    plt.title(title, fontdict=EXTRA_LARGE_FONTDICT)
    plt.plot(spec.detach().numpy(), '-o')
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def plot_spectra_X(X: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_spectra_generic(Z=X, title="$\\sigma_{p}(X)$", file=results_folder / "X_spectra.jpg")


def plot_spectra_fX(fX: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_spectra_generic(Z=fX, title="$\\sigma_{p}(f_{\star}(X))$", file=results_folder / "fX_spectra.jpg")


def plot_class_spectra_generic(Z: torch.Tensor, y: torch.Tensor, k: int,
                               title: str, smalltitles: typing.List[str], file: pathlib.Path):
    plt.rcParams['figure.figsize'] = 8, 8

    plt.suptitle(title, fontsize=EXTRA_LARGE_FONTDICT["fontsize"])
    for j in range(k):
        spec = spectrum(Z[y == j]).detach().numpy()
        plt.subplot(1, k, j + 1)
        plt.title(smalltitles[j], fontdict=LARGE_FONTDICT)
        plt.plot(spec, '-o')
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def plot_class_spectra_X(X: torch.Tensor, y: torch.Tensor, k: int, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_class_spectra_generic(Z=X, y=y, k=k,
                               title="$\\sigma_{p}(X_{j})$",
                               smalltitles=[f"$j={j + 1}$" for j in range(k)],
                               file=results_folder / "X_class_spectra.jpg"
                               )


def plot_class_spectra_fX(fX: torch.Tensor, y: torch.Tensor, k: int, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_class_spectra_generic(Z=fX, y=y, k=k,
                               title="$\\sigma_{p}(f_{\star}(X_{j}))$",
                               smalltitles=[f"$j={j + 1}$" for j in range(k)],
                               file=results_folder / "fX_class_spectra.jpg"
                               )


def plot_class_proj_residual_generic(Z1: torch.Tensor, Z2: torch.Tensor, y: torch.Tensor, k: int, title: str,
                                     smalltitles: typing.List[str], file: pathlib.Path):
    plt.rcParams['figure.figsize'] = 8, 8

    plt.suptitle(title, fontsize=EXTRA_LARGE_FONTDICT["fontsize"])
    for j in range(k):
        Z1j = Z1[y == j]  # (n_j, d)
        Z2j = Z2[y == j]  # (m_j, d)
        P_Col_Z2j = projection_onto_col(Z2j.T)  # (d, d)
        Z1j_proj = Z1j @ P_Col_Z2j  # (n_j, d)
        residuals = torch.linalg.norm(Z1j - Z1j_proj, dim=1)  # (n_j, )
        plt.subplot(1, k, j + 1)
        plt.title(smalltitles[j], fontdict=LARGE_FONTDICT)
        plt.hist(residuals.detach().numpy())
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


def plot_class_proj_residual_fX_fgfX(fX: torch.Tensor, fgfX: torch.Tensor, y: torch.Tensor, k: int,
                                     results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_class_proj_residual_generic(Z1=fX, Z2=fgfX, y=y, k=k,
                                     title="$\mathrm{Resid}(f_{\star}(X_{j}), \mathrm{Col}((f_{\star} \circ g_{\star} \circ f_{\star})(X_{j})))$",
                                     smalltitles=[f"$j = {j + 1}$" for j in range(k)],
                                     file=results_folder / "fX_fgfX_class_proj.jpg"
                                     )
