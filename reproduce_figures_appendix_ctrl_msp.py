from supervised_experiment import *


def reproduce_fig_d9():
    # Question: what is performance on correlated multiple-subspace data as a function of representation dimension?
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=20, d_S=[3, 4, 5])
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=30, d_S=[3, 4, 5])
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5])
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=60, d_S=[3, 4, 5])


def reproduce_fig_d10():
    # Question: what is performance on noisy correlated multiple-subspace data?
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5], nu=1e-6)
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5], nu=1e-4)
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5], nu=1e-2)
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5], nu=1)


def reproduce_fig_d11():
    # Question: what is comparison to different learning models on noisy correlated multiple-subspace data?
    ctrl_msp_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5])
    ctrl_msp_fcnn_experiment(n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5], d_latent=40, n_layers=2)
    infogan_experiment(n=[500, 500, 500], k=3, d_x=50, d_code=10, d_S=[3, 4, 5], d_noise=40, d_latent=40, n_layers=2)
    cgan_experiment(n=[500, 500, 500], k=3, d_x=50, d_S=[3, 4, 5], d_noise=40, d_latent=40, n_layers=2)
    cvae_experiment(n=[500, 500, 500], k=3, d_x=50, d_S=[3, 4, 5], d_z=40, d_latent=40, n_layers=2)


reproduce_fig_d9()
reproduce_fig_d10()
reproduce_fig_d11()
