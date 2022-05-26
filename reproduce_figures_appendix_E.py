from supervised_experiment import *


def reproduce_fig_12():
    # Question: what is default performance on benign data?
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_13():
    # Question: what happens when we increase number of classes on benign data?
    # Answer: same behavior, just with more classes
    ctrl_msp_experiment(
        n=[500, 500, 500, 500, 500], k=5, d_x=50, d_z=40, d_S=[3, 4, 5, 6, 7],
        nu=1e6, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_14():
    # Question: what is default performance on noisy benign data?
    # Answer: phase transition
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, eps_sq=1.0
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.025, eps_sq=1.0
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.05, eps_sq=1.0
    )


def reproduce_fig_15():
    # Question: what happens when eps decreases for benign data?
    # Answer: a little worse/more fuzzy, fails gracefully
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, eps_sq=1.0
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, eps_sq=0.75
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, eps_sq=0.5
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, eps_sq=0.25
    )


def reproduce_fig_16():
    # Question: what happens when we compare against other rep learning methods on benign data?
    # Answer: we do much better at learning subspaces
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, eps_sq=1.0
    )
    cgan_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_noise=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )
    infogan_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_noise=40, d_code=20, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )
    cvae_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e6, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )


def reproduce_fig_17():
    # Question: what is default performance on coherent data?
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_18():
    # Question: what happens when we increase number of classes on coherent data?
    # Answer: same behavior, just with more classes
    ctrl_msp_experiment(
        n=[500, 500, 500, 500, 500], k=5, d_x=50, d_z=40, d_S=[3, 4, 5, 6, 7],
        nu=0.1, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_19():
    # Question: what is default performance on noisy coherent data?
    # Answer: phase transition behavior
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=1.0
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.025, eps_sq=1.0
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.05, eps_sq=1.0
    )


def reproduce_fig_20():
    # Question: what happens when eps decreases for coherent data?
    # Answer: a little worse/more fuzzy, no realistic change
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=1.0
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=0.75
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=0.5
    )
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=0.25
    )


def reproduce_fig_21():
    # Question: what happens when we compare against other rep learning methods on coherent data?
    # Answer: we do much better at learning subspaces
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, eps_sq=1.0
    )
    cgan_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_noise=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )
    infogan_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_noise=40, d_code=20, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )
    cvae_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=0.1, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )


reproduce_fig_14()
reproduce_fig_15()
reproduce_fig_16()
reproduce_fig_17()
reproduce_fig_18()
reproduce_fig_19()
reproduce_fig_20()
reproduce_fig_21()
