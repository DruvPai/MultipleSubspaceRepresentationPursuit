from supervised_experiment import *


def reproduce_fig_1():
    # Question: what is baseline performance on correlated data?
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5]
    )


def reproduce_fig_2():
    # Question: what is performance on correlated data with added noise?
    ctrl_msp_experiment(
        n=[500, 500, 500], k=3, d_x=50, d_z=40, d_S=[3, 4, 5],
        nu=1e-2
    )


def reproduce_fig_3():
    # Question: what is default performance on real data?
    ctrl_msp_mnist_experiment(d_z=150)


def reproduce_fig_4():
    # Question: what is default performance on real data, when tuning with neural networks?
    ctrl_msp_fcnn_mnist_experiment(d_z=150, d_latent=150, n_layers=2)


#reproduce_fig_1()
#reproduce_fig_2()
#reproduce_fig_3()
reproduce_fig_4()
