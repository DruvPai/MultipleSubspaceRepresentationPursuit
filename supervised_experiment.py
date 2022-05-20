from data.multiple_subspaces import *
from model.ctrl_msp import *
from plotting.supervised import *
from plotting.ctrl_sg import *

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
    fX = model.F(X)
    fgfX = model.F(model.G(model.F((X))))

    print(torch.linalg.norm(X, dim=1))
    print(torch.linalg.norm(fX, dim=1))

    plot_spectra_X(X, result_dir)
    plot_spectra_fX(fX, result_dir)
    plot_class_spectra_X(X, y, k, result_dir)
    plot_class_spectra_fX(fX, y, k, result_dir)
    plot_class_proj_residual_fX_fgfX(fX, fgfX, y, k, result_dir)
    plot_cosine_similarity_X(X, result_dir)
    plot_cosine_similarity_fX(fX, result_dir)
    if isinstance(model, SupervisedCTRLSG):
        plot_E_C(model, result_dir)


def ctrl_msp_experiment(n: typing.List[int], k: int, d_x: int, d_z: int, d_S: typing.List[int], nu: float = 0.1,
                        sigma_sq: float = 0.0, eps_sq: float = 1.0,
                        lr_f: float = 1e-2, lr_g: float = 1e-3, inner_opt_steps: int = 1000,
                        outlier_pct: float = 0.0, outlier_mag: float = 0.0, label_corruption_pct: float = 0.0,
                        batch_size: int = 50, epochs: int = 2):
    pl.utilities.seed.reset_seed()
    data = MultipleSubspacesDataModule(n, k, d_x, d_S, nu, sigma_sq, batch_size,
                                       outlier_pct, outlier_mag, label_corruption_pct)
    model = CTRLMSP(d_x, d_z, eps_sq, lr_f, lr_g, inner_opt_steps)
    supervised_experiment(model, data, epochs)
