import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
N = 100
tau = 1.0
t = np.linspace(0, 100, 2000)
g_values = [1.1, 0.99]  
num_simulations = 10

color_palette = ['#5B8A3A', '#7A5B9D']  

def phi(x):
    return np.tanh(x)

def network_dynamics(r, t, J, tau):
    return (-r + np.dot(J, phi(r))) / tau

# Set up the plot
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(bottom=0.1, hspace=0.3, wspace=0.3)

for g_idx, g in enumerate(g_values):
    color = color_palette[g_idx]
    
    for sim in range(num_simulations):
        # Simulate network
        J = np.random.normal(0, g*g/np.sqrt(N), size=(N, N))
        np.fill_diagonal(J, 0)
        r0 = np.random.rand(N) * 0.1
        sol = odeint(network_dynamics, r0, t, args=(J, tau))

        trajectories = 20
        # Time series plot
        for i in range(trajectories):
            axs[0, 0].plot(t, sol[:, i], color=color, alpha=(1 - i / trajectories) * 0.5)
        
        # Phase plot with direction and units
        for i in range(len(sol) - 1):
            axs[0, 1].plot(sol[i:i+2, 0], sol[i:i+2, 1], color=color, alpha=(1 - i / len(sol)) * 0.5, linewidth=1)
        
        # Mean activity plot
        mean_activity = np.mean(sol, axis=1)
        axs[1, 0].plot(t, mean_activity, color=color, alpha=0.5)

        # Autocorrelation plot
        autocorr = np.correlate(mean_activity - np.mean(mean_activity), 
                                mean_activity - np.mean(mean_activity), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        axs[1, 1].plot(t[:len(autocorr)], autocorr, color=color, alpha=0.5)


axs[0, 0].set_xlabel(r'Time', fontsize=14)
axs[0, 0].set_ylabel(r'Activity', fontsize=14)
axs[0, 0].set_title(r'Network Dynamics', fontsize=16)
axs[0, 0].axhline(y=0, color='black', linestyle='--')
axs[0, 0].grid(True)
axs[0, 0].set_ylim(-2, 2)
axs[0, 0].set_xlim(0, 100)

axs[0, 1].set_xlabel(r'Unit 1', fontsize=14)
axs[0, 1].set_ylabel(r'Unit 2', fontsize=14)
axs[0, 1].set_title(r'Phase Plot', fontsize=16)
axs[0, 1].grid(True)

axs[1, 0].set_xlabel(r'Time', fontsize=14)
axs[1, 0].set_ylabel(r'Mean Activity', fontsize=14)
axs[1, 0].set_title(r'Mean Network Activity', fontsize=16)
axs[1, 0].set_xlim(0, 100)
axs[1, 0].grid(True)

axs[1, 1].set_xlabel(r'Time Lag', fontsize=14)
axs[1, 1].set_ylabel(r'Autocorrelation', fontsize=14)
axs[1, 1].set_title(r'Autocorrelation of Mean Activity', fontsize=16)
axs[1, 1].set_xlim(0, 100)
axs[1, 1].grid(True)

legend_elements = [plt.Line2D([0], [0], color=color_palette[0], lw=2, label=f'g = {g_values[0]}'),
                   plt.Line2D([0], [0], color=color_palette[1], lw=2, label=f'g = {g_values[1]}')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)

plt.show()
