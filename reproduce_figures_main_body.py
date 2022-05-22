from supervised_experiment import *


def reproduce_fig_1():
    # Question: what is default performance on benign data?
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_2():
    # Question: what is default performance on correlated data?
    # Answer: same as in incoherent data.
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_3():
    # Question: what happens when we introduce noise to correlated data?
    # Answer: learn the same structures.
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=1.0
    )


def reproduce_fig_4():
    # Question: how does this compare to supervised rep. learning methods?
    # Answer: much better
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=1.0
    )
    cgan_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_noise=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )
    cvae_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )
    infogan_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_code=15, d_noise=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )


reproduce_fig_1()
reproduce_fig_2()
reproduce_fig_3()
reproduce_fig_4()
