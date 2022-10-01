from unsupervised_experiment import *


def reproduce_fig_c6():
    # Question: what is baseline performance on single-subspace data?
    ctrl_ssp_experiment(n=500, d_x=50, d_z=40, d_S=10)


def reproduce_fig_c7():
    # Question: what is performance on single-subspace data as a function of representation dimension?
    ctrl_ssp_experiment(n=500, d_x=50, d_z=10, d_S=10)
    ctrl_ssp_experiment(n=500, d_x=50, d_z=20, d_S=10)
    ctrl_ssp_experiment(n=500, d_x=50, d_z=40, d_S=10)
    ctrl_ssp_experiment(n=500, d_x=50, d_z=50, d_S=10)


def reproduce_fig_c8():
    # Question: what is performance on noisy single-subspace data?
    ctrl_ssp_experiment(n=500, d_x=50, d_z=40, d_S=10, nu=1e-6)
    ctrl_ssp_experiment(n=500, d_x=50, d_z=40, d_S=10, nu=1e-4)
    ctrl_ssp_experiment(n=500, d_x=50, d_z=40, d_S=10, nu=1e-2)
    ctrl_ssp_experiment(n=500, d_x=50, d_z=40, d_S=10, nu=1)


reproduce_fig_c6()
reproduce_fig_c7()
reproduce_fig_c8()
