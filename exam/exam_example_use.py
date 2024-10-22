from exam_functions import *
from exam_script import *
import numpy as np
import random
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

"""
Maps: Bifurcations and Lyapunov exponents
"""
# # Tent Map
# plot_bifurcation(tent_map, (0, 2), 0.1)
# lyap_tent = calculate_lyapunov(tent_map, 0.1, params=(1.99,))
# print(f"Lyapunov exponent for tent map: {lyap_tent}")

# # Logistic Map
# plot_bifurcation(logistic_map, (0, 4), 0.1)
# lyap_logistic = calculate_lyapunov(logistic_map, 0.1, params=(3.9,))
# print(f"Lyapunov exponent for logistic map: {lyap_logistic}")

# plot_bifurcation(henon_map, (1, 1.5), [0.1, 0.1], other_params=(0.3,))
# lyap_henon = calculate_lyapunov(henon_map, [0.1, 0.1], params=(1.4, 0.3))
# print(f"Lyapunov exponents for Henon map: {lyap_henon}")




""" 
Dynamic System Analysis
"""
# x0 = [1, 1, 1]
# t = np.linspace(0, 100, 10000)

# solution = odeint(lorenz_system, x0, t)

# plot_time_series(lorenz_system, x0, 100, discrete=False)


# # Lorenz system
# poincare_lorenz = poincare_section(lorenz_system, x0, 10000, 2, 27)  # z = 27 plane
# print(f"Lorenz Poincaré section shape: {poincare_lorenz.shape}")
# print(f"Lorenz Poincaré section size: {poincare_lorenz.size}")
# plot_poincare_section(poincare_lorenz)

# # 3D plot of Lorenz system
# plot_3d_system(lorenz_system, x0, (0, 100))




""" 
Entropy and Dimension Calculation
"""

# x0 = [1, 1, 1]
# t = np.linspace(0, 100, 10000)

# solution = odeint(lorenz_system, x0, t)

# attractor = solution[:, :3]
# r_range = np.logspace(-2, 1, 20)
# corr_dim = calculate_correlation_dimension(attractor, r_range)
# print(f"Correlation Dimension: {corr_dim}")

# entropy = calculate_entropy(solution[:, 0])
# print(f"Entropy: {entropy}")


"""
Tent map entropy (Sequence Entropy)
"""
# mu = 1.99  # Parameter for the tent map
# N = 5000  # Number of iterations
# sigma = np.zeros(N)
# x = np.random.uniform()  # Initial condition

# for t in range(N):
#     x = tent_map(x, mu)
#     sigma[t] = (x > 0.5)

# p = (sigma @ sigma) / N
# S = -p * np.log(p) - (1-p) * np.log(1-p)

# print(f"Entropy S = {S:.6f}")
# print(f"S=log({np.exp(S):.6f})")




"""
Recurrence Analysis
"""

# x0 = [1, 1, 1]
# t = np.linspace(0, 100, 10000)

# solution = odeint(lorenz_system, x0, t)


# threshold = 2.0
# recurrence_plot = plot_recurrence(solution[:, 0], threshold)
# # rqa_measures = recurrence_quantification_analysis(recurrence_plot)
# # print("RQA Measures:", rqa_measures)




