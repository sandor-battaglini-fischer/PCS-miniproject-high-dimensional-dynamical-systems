import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

N = 1000
tau = 1.0
t = np.linspace(0, 500, 2000)

def phi(x):
    return np.tanh(x)

def network_dynamics(r, t, J, tau):
    return (-r + np.dot(J, phi(r))) / tau

def simulate_network(g):
    J = np.random.normal(0, g*g/np.sqrt(N), size=(N, N))
    np.fill_diagonal(J, 0)
    
    r0 = np.random.rand(N) * 0.1
    solution = odeint(network_dynamics, r0, t, args=(J, tau))
    
    return solution, J

def calculate_spectral_radius(J):
    eigenvalues = np.linalg.eigvals(J)
    return np.max(np.abs(eigenvalues))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

colors = plt.cm.viridis(np.linspace(0, 1, 5))

g_values = [0.99, 1.1]

 
color_g_0_99 = '#7A5B9D'
color_g_1_5 = '#5B8A3A'

for i, g in enumerate(g_values):
    sol, J = simulate_network(g)
    spectral_radius = calculate_spectral_radius(J)
    
    eigenvalues = np.linalg.eigvals(J)
    color = color_g_0_99 if g == 0.99 else color_g_1_5
    axs[i, 0].scatter(eigenvalues.real, eigenvalues.imag, alpha=0.5, c=color)
    axs[i, 0].set_xlim(-1.5, 1.5)
    axs[i, 0].set_ylim(-1.5, 1.5)
    axs[i, 0].set_aspect('equal')
    axs[i, 0].set_title(r'$\text{Coupling Constant } g = %.2f$' % g, fontsize=16)
    axs[i, 0].set_xlabel(r'Re($\lambda$)', fontsize=14)
    axs[i, 0].set_ylabel(r'Im($\lambda$)', fontsize=14)
    
    axs[i, 0].axvline(x=1, color='red', linestyle='--', label='Spectral Radius = 1')
    axs[i, 0].axvline(x=-1, color='red', linestyle='--')

    for j in range(10):
        axs[i, 1].plot(t, sol[:, j], color=color)
    axs[i, 1].set_xlim(0, 100)
    axs[i, 1].set_ylim(-1, 1)
    axs[i, 1].set_title(r'$\text{Trajectories for} \ g = %.2f$' % g, fontsize=16)
    axs[i, 1].set_xlabel(r'$t$', fontsize=14)
    axs[i, 1].set_ylabel(r'r$_i$', fontsize=14)
    
    axs[i, 1].axhline(y=0, color='black', linestyle='--')
    

plt.tight_layout()
plt.show()
