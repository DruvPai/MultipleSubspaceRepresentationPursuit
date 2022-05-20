from data.single_subspace import *
from model.ctrl_ssp import *
from plotting.unsupervised import *
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
    X = data.X_test
    fX = model.f(X)
    fgfX = model.fgf(X)

    plot_isometry_X_fX(X, fX, result_dir)
    plot_proj_residual_fX_fgfX(fX, fgfX, result_dir)
    if isinstance(model, UnsupervisedCTRLSG):
        plot_E_C(model, result_dir)


def ctrl_ssp_experiment(n: int, d_x: int, d_z: int, d_S: int, sigma_sq: float = 0.0, eps_sq: float = 1.0,
                        lr_f: float = 1e-2, lr_g: float = 1e-3, inner_opt_steps: int = 1000,
                        outlier_pct: float = 0.0, outlier_mag: float = 0.0,
                        batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = SingleSubspaceDataModule(n, d_x, d_S, sigma_sq, batch_size, outlier_pct, outlier_mag)
    model = CTRLSSP(d_x, d_z, eps_sq, lr_f, lr_g, inner_opt_steps)
    unsupervised_experiment(model, data, epochs)
