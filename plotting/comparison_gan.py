import pathlib

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from plotting.supervised import *


def plot_cosine_similarity_gZ(gZ: torch.Tensor, results_folder: pathlib.Path):
    results_folder.mkdir(parents=True, exist_ok=True)
    plot_cosine_similarity_generic(
        Z=gZ,
        title="$\cos(\\angle(g_{\star}(z^i), g_{\star}(z^j)))$",
        file=results_folder / "gZ_heatmap.jpg"
    )
