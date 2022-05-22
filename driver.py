from supervised_experiment import *

# ctrl_ssp_experiment(500, 50, 40, 10, 0.0, batch_size=50, epochs=5)
# ctrl_ssp_experiment(500, 50, 40, 40, 0.0, batch_size=50, epochs=5)
# ctrl_ssp_experiment(500, 30, 40, 10, 0.0, batch_size=50, epochs=5)
# ctrl_msp_experiment([500, 500, 500], 3, 50, 40, [3, 4, 5], nu=1e6, sigma_sq=0.0, batch_size=50, epochs=2)
cvae_experiment([500, 500, 500], 3, 45, 40, [3, 4, 5], epochs=100)
