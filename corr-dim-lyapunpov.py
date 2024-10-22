import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def phi(x):
    return np.tanh(x)

def network_dynamics(r, t, J):
    return -r + np.dot(J, phi(r))

def calculate_lyapunov_spectrum(g, N, t_sim, dt):
    J = np.random.normal(0, g*g/np.sqrt(N), size=(N, N))
    np.fill_diagonal(J, 0)
    
    t = np.arange(0, t_sim, dt)
    r0 = np.random.randn(N) * 0.1
    
    def jacobian(r):
        return -np.eye(N) + J * (1 - phi(r)**2)
    
    trajectory = odeint(network_dynamics, r0, t, args=(J,))
    
    Q = np.eye(N)
    lyap = np.zeros(N)
    
    for i in range(1, len(t)):
        J_t = jacobian(trajectory[i])
        Q_new = np.dot(J_t, Q)
        Q, R = np.linalg.qr(Q_new)
        lyap += np.log(np.abs(np.diag(R)))
    
    lyap /= t_sim
    return np.sort(lyap)[::-1]


N = 100  
t_sim = 100  
dt = 0.1
g_values = np.logspace(0, 3, 100)

lyap_spectra = []
entropy_rates = []
attractor_dims = []

for g in g_values:
    lyap = calculate_lyapunov_spectrum(g, N, t_sim, dt)
    lyap_spectra.append(lyap)
    
    positive_lyap = lyap[lyap > 0]
    entropy_rates.append(np.sum(positive_lyap))
    
    k = np.argmax(np.cumsum(lyap) < 0)
    attractor_dims.append(k + np.sum(lyap[:k]) / np.abs(lyap[k]))

plt.figure(figsize=(15, 10))

# Plot (a)
plt.subplot(221)
g_values_a = [1000, 100, 10, 1]
colors = ['red', 'purple', 'blue', 'green']
for g, color in zip(g_values_a, colors):
    lyap = calculate_lyapunov_spectrum(g, N, t_sim, dt)
    plt.plot(range(1, N+1), lyap, color=color, label=f'g = {g}')
plt.xscale('log')
plt.ylim(-0.5, 3)
plt.xlabel('i')
plt.ylabel('λ (1/τ)')
plt.title('(a)')
plt.legend()

# Plot (b)
plt.subplot(222)
plt.semilogx(g_values, [spec[0] for spec in lyap_spectra], color='orange')
plt.xlabel('g')
plt.ylabel('λmax (1/τ)')
plt.title('(b)')
plt.axhline(y=0, color='k', linestyle='--')

# Plot (c)
plt.subplot(223)
plt.semilogx(g_values, entropy_rates, color='orange')
plt.xlabel('g')
plt.ylabel('H/(N/τ)')
plt.title('(c)')

# Plot (d)
plt.subplot(224)
plt.semilogx(g_values, [d/N*100 for d in attractor_dims], color='orange')
plt.xlabel('g')
plt.ylabel('D/N(%)')
plt.title('(d)')

plt.tight_layout()
plt.show()
