import pathlib

import matplotlib.pyplot as plt
import pytorch_lightning as pl

EXTRA_LARGE_FONTDICT = {"fontsize": 34}
LARGE_FONTDICT = {"fontsize": 30}


def plot_Q_C(model: pl.LightningModule, results_folder: pathlib.Path):
    plt.rcParams['figure.figsize'] = 8, 8

    results_folder.mkdir(parents=True, exist_ok=True)

    training_Q = model.training_Q
    training_C = model.training_C

    plt.title("Training Losses", fontdict=EXTRA_LARGE_FONTDICT)
    plt.plot(training_Q, label="$\mathcal{Q}(f)$")
    plt.plot(training_C, label="$\mathcal{C}(f, g)$")
    plt.legend()
    plt.savefig(results_folder / "training_Q_C.jpg")
    plt.close()
