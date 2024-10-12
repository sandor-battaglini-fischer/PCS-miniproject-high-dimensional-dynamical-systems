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

N = 1000
t_sim = 100
dt = 0.1
g_values = np.linspace(0.5, 2.0, 20)

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

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(g_values, [spec[0] for spec in lyap_spectra])
plt.xlabel('g')
plt.ylabel('Largest Lyapunov exponent')
plt.title('(b)')

plt.subplot(132)
plt.plot(g_values, entropy_rates)
plt.xlabel('g')
plt.ylabel('Entropy rate H')
plt.title('(c)')

plt.subplot(133)
plt.plot(g_values, [d/N for d in attractor_dims])
plt.xlabel('g')
plt.ylabel('Relative attractor dim. D/N')
plt.title('(d)')

plt.tight_layout()
plt.show()