import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider

# Parameters
N = 100
tau = 1.0
t = np.linspace(0, 100, 2000)

def phi(x):
    return np.tanh(x)

def network_dynamics(r, t, J, tau):
    return (-r + np.dot(J, phi(r))) / tau

def simulate_network(g):
    J = np.random.normal(0, g*g/np.sqrt(N), size=(N, N))
    np.fill_diagonal(J, 0)  
    
    r0 = np.random.rand(N) * 0.1
    solution = odeint(network_dynamics, r0, t, args=(J, tau))
    
    return solution


fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(bottom=0.1, hspace=0.3, wspace=0.3)

g_init = 1.0

# Time series plot
ax_time = axs[0, 0]
sol = simulate_network(g_init)
lines_time = ax_time.plot(t, sol[:, :5])
ax_time.set_xlabel('Time')
ax_time.set_ylabel('Activity')
ax_time.set_title(f'Network Dynamics (g = {g_init:.2f})')

# Phase plot
ax_phase = axs[0, 1]
lines_phase = ax_phase.plot(sol[:, 0], sol[:, 1])
ax_phase.set_xlabel('Unit 1')
ax_phase.set_ylabel('Unit 2')
ax_phase.set_title('Phase Plot')

# Mean activity plot
ax_mean = axs[1, 0]
mean_activity = np.mean(sol, axis=1)
line_mean, = ax_mean.plot(t, mean_activity)
ax_mean.set_xlabel('Time')
ax_mean.set_ylabel('Mean Activity')
ax_mean.set_title('Mean Network Activity')

# Autocorrelation plot
ax_autocorr = axs[1, 1]
autocorr = np.correlate(mean_activity - np.mean(mean_activity), 
                        mean_activity - np.mean(mean_activity), mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr /= autocorr[0]
line_autocorr, = ax_autocorr.plot(t[:len(autocorr)], autocorr)
ax_autocorr.set_xlabel('Time Lag')
ax_autocorr.set_ylabel('Autocorrelation')
ax_autocorr.set_title('Autocorrelation of Mean Activity')

# Add slider
ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
slider = Slider(ax_slider, 'g', 0.5, 1.5, valinit=g_init, valstep=0.01)

def update(val):
    g = slider.val
    sol = simulate_network(g)
    
    # Update time series plot
    for i, line in enumerate(lines_time):
        line.set_ydata(sol[:, i])
    ax_time.set_title(f'Network Dynamics (g = {g:.2f})')
    
    # Update phase plot
    lines_phase[0].set_data(sol[:, 0], sol[:, 1])
    ax_phase.relim()
    ax_phase.autoscale_view()
    
    # Update mean activity plot
    mean_activity = np.mean(sol, axis=1)
    line_mean.set_ydata(mean_activity)
    ax_mean.relim()
    ax_mean.autoscale_view()
    
    # Update autocorrelation plot
    autocorr = np.correlate(mean_activity - np.mean(mean_activity), 
                            mean_activity - np.mean(mean_activity), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0]
    line_autocorr.set_ydata(autocorr)
    ax_autocorr.relim()
    ax_autocorr.autoscale_view()
    
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
