from data.single_subspace import *
from model.comparisons.gan import *
from model.comparisons.vae import *
from model.ctrl_ssp import *
from plotting.comparison import *
from plotting.ctrl_sg import *

UNSUPERVISED_EXPERIMENT_DIR = pathlib.Path("experiments") / "unsupervised"

pl.seed_everything(1337)


def unsupervised_experiment(model: pl.LightningModule, data: pl.LightningDataModule, epochs: int = 2):
    pl.utilities.seed.reset_seed()

    result_dir = UNSUPERVISED_EXPERIMENT_DIR / data.name / model.name / f"e{epochs}"
    result_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, data)

    # plotting
    if isinstance(model, UnsupervisedCTRLSG):
        X = data.X_test
        fX = model.f(X)
        gfX = model.gf(X)
        fgfX = model.fgf(X)

        plot_spectra_X(X, result_dir)
        plot_spectra_fX(fX, result_dir)
        plot_spectra_gfX(gfX, result_dir)
        plot_spectra_fgfX(fgfX, result_dir)
        plot_isometry_X_fX(X, fX, result_dir)
        plot_proj_residual_X_gfX(X, gfX, result_dir)
        plot_proj_residual_fX_fgfX(fX, fgfX, result_dir)
        plot_E_C(model, result_dir)
    elif isinstance(model, UnsupervisedVAE):
        X = data.X_test
        gfX = model.autoencode_data(X)
        plot_spectra_X(X, result_dir)
        plot_spectra_gfX(gfX, result_dir)
        plot_proj_residual_X_gfX(X, gfX, result_dir)
    elif isinstance(model, UnsupervisedGAN):
        X = data.X_test
        gZ = model.generate_data(X.shape[0])
        plot_spectra_X(X, result_dir)
        plot_spectra_gZ(gZ, result_dir)
        plot_proj_residual_X_gZ(X, gZ, result_dir)


def ctrl_ssp_experiment(n: int, d_x: int, d_z: int, d_S: int, sigma_sq: float = 0.0,
                        outlier_pct: float = 0.0, outlier_mag: float = 0.0,
                        eps_sq: float = 1.0, lr_f: float = 1e-2, lr_g: float = 1e-3,
                        inner_opt_steps: int = 1000, batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = SingleSubspaceDataModule(n, d_x, d_S, sigma_sq, outlier_pct, outlier_mag, batch_size)
    model = CTRLSSP(d_x, d_z, eps_sq, lr_f, lr_g, inner_opt_steps)
    unsupervised_experiment(model, data, epochs)


def vanillavae_experiment(n: int, d_x: int, d_z: int, d_S: int, sigma_sq: float = 0.0, outlier_pct: float = 0.0,
                          outlier_mag: float = 0.0, n_layers: int = 5, d_latent: int = 50, lr: float = 1e-4,
                          batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = SingleSubspaceDataModule(n, d_x, d_S, sigma_sq, outlier_pct, outlier_mag, batch_size)
    model = UnsupervisedVanillaVAE(d_x, d_z, d_latent, n_layers, lr)
    unsupervised_experiment(model, data, epochs)


def vanillagan_experiment(n: int, d_x: int, d_noise: int, d_S: int, sigma_sq: float = 0.0, outlier_pct: float = 0.0,
                          outlier_mag: float = 0.0, n_layers: int = 5, d_latent: int = 50, lr: float = 1e-4,
                          batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = SingleSubspaceDataModule(n, d_x, d_S, sigma_sq, outlier_pct, outlier_mag, batch_size)
    model = UnsupervisedVanillaGAN(d_x, d_noise, d_latent, n_layers, lr)
    unsupervised_experiment(model, data, epochs)
