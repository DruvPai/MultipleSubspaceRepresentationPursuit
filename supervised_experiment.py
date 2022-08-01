from data.multiple_subspaces import *
from data.mnist import *
from model.comparisons.gan import *
from model.comparisons.vae import *
from model.ctrl_msp import *
from plotting.comparison import *
from plotting.ctrl_sg import *
from plotting.supervised import *


SUPERVISED_EXPERIMENT_DIR = pathlib.Path("experiments") / "supervised"

pl.seed_everything(1337)


def supervised_experiment(model: pl.LightningModule, data: pl.LightningDataModule, epochs: int = 2):
    pl.utilities.seed.reset_seed()

    result_dir = SUPERVISED_EXPERIMENT_DIR / data.name / model.name / f"e{epochs}"
    result_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, data)

    # plotting
    X = data.X_test
    y = data.y_test
    k = data.k

    if isinstance(model, SupervisedCTRLSG):
        fX = model.f(X)
        gfX = model.gf(X)
        fgfX = model.fgf(X)

        plot_spectra_X(X, result_dir)
        plot_spectra_fX(fX, result_dir)
        plot_spectra_gfX(gfX, result_dir)
        plot_spectra_fgfX(fgfX, result_dir)
        plot_class_spectra_X(X, y, k, result_dir)
        plot_class_spectra_fX(fX, y, k, result_dir)
        plot_class_spectra_gfX(gfX, y, k, result_dir)
        plot_class_spectra_fgfX(fgfX, y, k, result_dir)
        plot_class_proj_residual_X_gfX(X, gfX, y, k, result_dir)
        plot_class_proj_residual_fX_fgfX(fX, fgfX, y, k, result_dir)
        plot_cosine_similarity_X(X, result_dir)
        plot_cosine_similarity_fX(fX, result_dir)
        plot_cosine_similarity_gfX(gfX, result_dir)
        plot_cosine_similarity_fgfX(fgfX, result_dir)
        plot_E_C(model, result_dir)

    elif isinstance(model, SupervisedGAN):
        gZ = model.generate_data(X.shape[0], y)

        plot_spectra_X(X, result_dir)
        plot_spectra_gZ(gZ, result_dir)
        plot_class_spectra_X(X, y, k, result_dir)
        plot_class_spectra_gZ(gZ, y, k, result_dir)
        plot_class_proj_residual_X_gZ(X, gZ, y, k, result_dir)
        plot_cosine_similarity_X(X, result_dir)
        plot_cosine_similarity_gZ(gZ, result_dir)

    elif isinstance(model, SupervisedVAE):
        gfX = model.autoencode_data(X, y)

        plot_spectra_X(X, result_dir)
        plot_spectra_gfX(gfX, result_dir)
        plot_class_spectra_X(X, y, k, result_dir)
        plot_class_spectra_gfX(gfX, y, k, result_dir)
        plot_class_proj_residual_X_gfX(X, gfX, y, k, result_dir)
        plot_cosine_similarity_X(X, result_dir)
        plot_cosine_similarity_gfX(gfX, result_dir)


def ctrl_msp_experiment(n: typing.List[int], k: int, d_x: int, d_z: int, d_S: typing.List[int], nu: float = 0.1,
                        sigma_sq: float = 0.0, eps_sq: float = 1.0,
                        lr_f: float = 1e-2, lr_g: float = 1e-3, inner_opt_steps: int = 1000,
                        outlier_pct: float = 0.0, outlier_mag: float = 0.0, label_corruption_pct: float = 0.0,
                        batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = MultipleSubspacesDataModule(n, k, d_x, d_S, nu, sigma_sq, outlier_pct, outlier_mag, label_corruption_pct,
                                       batch_size)
    model = CTRLMSP(d_x, d_z, eps_sq, lr_f, lr_g, inner_opt_steps)
    supervised_experiment(model, data, epochs)


def infogan_experiment(n: typing.List[int], k: int, d_x: int, d_noise: int, d_code: int, d_S: typing.List[int],
                       nu: float = 0.1, sigma_sq: float = 0.0, d_latent: int = 50, n_layers: int = 5,
                       lr: float = 1e-4, outlier_pct: float = 0.0, outlier_mag: float = 0.0,
                       label_corruption_pct: float = 0.0, batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = MultipleSubspacesDataModule(n, k, d_x, d_S, nu, sigma_sq, outlier_pct, outlier_mag, label_corruption_pct,
                                       batch_size)
    model = SupervisedInfoGAN(k, d_x, d_noise, d_code, d_latent, n_layers, lr)
    supervised_experiment(model, data, epochs)


def cgan_experiment(n: typing.List[int], k: int, d_x: int, d_noise: int, d_S: typing.List[int],
                    nu: float = 0.1, sigma_sq: float = 0.0, d_latent: int = 50, n_layers: int = 5,
                    lr: float = 1e-4, outlier_pct: float = 0.0, outlier_mag: float = 0.0,
                    label_corruption_pct: float = 0.0, batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = MultipleSubspacesDataModule(n, k, d_x, d_S, nu, sigma_sq, outlier_pct, outlier_mag, label_corruption_pct,
                                       batch_size)
    model = SupervisedCGAN(k, d_x, d_noise, d_latent, n_layers, lr)
    supervised_experiment(model, data, epochs)


def cvae_experiment(n: typing.List[int], k: int, d_x: int, d_z: int, d_S: typing.List[int],
                    nu: float = 0.1, sigma_sq: float = 0.0, d_latent: int = 50, n_layers: int = 5,
                    lr: float = 1e-4, outlier_pct: float = 0.0, outlier_mag: float = 0.0,
                    label_corruption_pct: float = 0.0, batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = MultipleSubspacesDataModule(n, k, d_x, d_S, nu, sigma_sq, outlier_pct, outlier_mag, label_corruption_pct,
                                       batch_size)
    model = SupervisedCVAE(k, d_x, d_z, d_latent, n_layers, lr)
    supervised_experiment(model, data, epochs)

def ctrl_msp_mnist_experiment(d_z: int, eps_sq: float = 1.0, lr_f: float = 1e-2, lr_g: float = 1e-3,
                              inner_opt_steps: int = 100, batch_size: int = 50, epochs: int = 1):
    pl.utilities.seed.reset_seed()
    data = MNISTDataModule(data_dir="./datasets/", val_split=0.0, normalize=False, flatten=True, batch_size=batch_size)
    model = CTRLMSP(data.unrolled_dim, d_z, eps_sq, lr_f, lr_g, inner_opt_steps)
    supervised_experiment(model, data, epochs)

def ctrl_msp_fcnn_mnist_experiment(d_z: int, d_latent: int, n_layers: int, eps_sq: float = 1.0,
                             lr_f: float = 1e-2, lr_g: float = 1e-3,
                             inner_opt_steps: int = 100, batch_size: int = 50, epochs: int = 1):
    pl.utilities.seed.reset_seed()
    data = MNISTDataModule(data_dir="./datasets/", val_split=0.0, normalize=False, flatten=True, batch_size=batch_size)
    model = CTRLMSPFCNN(data.unrolled_dim, d_z,  d_latent, n_layers, eps_sq, lr_f, lr_g, inner_opt_steps)
    supervised_experiment(model, data, epochs)
