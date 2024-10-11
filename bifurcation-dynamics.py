import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider

# Parameters
N = 100
tau = 1.0  
t = np.linspace(0, 500, 1000)  

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

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)  

g_init = 1.0
sol = simulate_network(g_init)
lines = ax.plot(t, sol[:, :5])
ax.set_xlabel('Time')
ax.set_ylabel('Activity')
ax.set_title(f'Network Dynamics (g = {g_init:.2f})')

ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
slider = Slider(ax_slider, 'g', 0.5, 1.5, valinit=g_init, valstep=0.01)

def update(val):
    g = slider.val
    sol = simulate_network(g)
    for i, line in enumerate(lines):
        line.set_ydata(sol[:, i])
    ax.set_title(f'Network Dynamics (g = {g:.2f})')
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
