from unsupervised_experiment import *


def reproduce_fig_5():
    # Question: what is default performance?
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_6():
    # Question: what happens when X dimension is larger?
    # Answer: nothing significant changes, default performance
    ctrl_ssp_experiment(
        n=500, d_x=80, d_z=40, d_S=10, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_7():
    # Question: what happens when Z dimension is larger?
    # Answer: the orthogonal map is a perfect encoding
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=60, d_S=10, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_8():
    # Question: what happens when S dimension is larger?
    # Answer: nothing significant changes, default performance
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=20, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_9():
    # Question: what happens when S dimension is larger?
    # Answer: nothing significant changes, default performance
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=40, sigma_sq=0.0, eps_sq=1.0
    )


def reproduce_fig_10():
    # Question: what happens with noise?
    # Answer: performance maintains or decays gracefully
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.01, eps_sq=1.0
    )


def reproduce_fig_11():
    # Question: what happens with noise and reduced epsilon?
    # Answer: performance maintains or decays gracefully
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.01, eps_sq=0.75
    )
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.01, eps_sq=0.5
    )
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.01, eps_sq=0.25
    )


def reproduce_fig_12():
    # Question: what happens with increased noise?
    # Answer: performance maintains or decays gracefully
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.025, eps_sq=1.0
    )
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.05, eps_sq=1.0
    )
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.075, eps_sq=1.0
    )
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.1, eps_sq=1.0
    )


def reproduce_fig_13():
    # Question: can we compare to other representation learning methods?
    # Answer: we do much better at recovering subspaces.
    ctrl_ssp_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.01, eps_sq=1.0
    )
    vanillagan_experiment(
        n=500, d_x=50, d_noise=40, d_S=10, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )
    vanillavae_experiment(
        n=500, d_x=50, d_z=40, d_S=10, sigma_sq=0.01, d_latent=100, n_layers=10, epochs=1000
    )


reproduce_fig_5()
reproduce_fig_6()
reproduce_fig_7()
reproduce_fig_8()
reproduce_fig_9()
reproduce_fig_10()
reproduce_fig_11()
reproduce_fig_12()
reproduce_fig_13()
