from plotting.supervised import *
from plotting.unsupervised import *


def plot_cosine_similarity_gZ(gZ: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_cosine_similarity_generic(
        Z=gZ,
        title="$\cos(\\angle(g_{\star}(z^i), g_{\star}(z^j)))$",
        file=results_folder / "gZ_heatmap.jpg"
    )


def plot_spectra_gZ(gZ: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_spectra_generic(Z=gZ, title="$\\sigma_{p}(g_{\star}(Z))$", file=results_folder / "gZ_spectra.jpg")


def plot_class_spectra_gZ(gZ: torch.Tensor, y: torch.Tensor, k: int, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_class_spectra_generic(Z=gZ, y=y, k=k,
                               title="$\\sigma_{p}(g_{\star}(Z_{j}))$",
                               smalltitles=[f"$j={j + 1}$" for j in range(k)],
                               file=results_folder / "gZ_class_spectra.jpg"
                               )


def plot_proj_residual_X_gZ(X: torch.Tensor, gZ: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_proj_residual_generic(Z1=X, Z2=gZ,
                               title="$\mathrm{Resid}(X, \mathrm{Col}(g_{\star}(Z)))$",
                               file=results_folder / "X_gZ_proj.jpg")


def plot_class_proj_residual_X_gZ(X: torch.Tensor, gZ: torch.Tensor, y: torch.Tensor, k: int,
                                  results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_class_proj_residual_generic(Z1=X, Z2=gZ, y=y, k=k,
                                     title="$\mathrm{Resid}(X_{j}, \mathrm{Col}(g_{\star}(Z_{j})))$",
                                     smalltitles=[f"$j = {j + 1}$" for j in range(k)],
                                     file=results_folder / "X_gZ_class_proj.jpg"
                                     )
