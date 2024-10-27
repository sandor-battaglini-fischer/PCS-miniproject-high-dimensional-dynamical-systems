import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm, qr
from tqdm import tqdm  # For progress bar

def phi(x):
    return np.tanh(x)

def network_dynamics(t, r, J):
    return -r + np.dot(J, phi(r))

def calculate_lyapunov_spectrum(g, N, t_sim, dt):
    J = np.random.normal(0, g*g/np.sqrt(N), size=(N, N))
    np.fill_diagonal(J, 0)
    
    t_eval = np.arange(0, t_sim, dt)
    r0 = np.random.randn(N) * 0.1
    
    def jacobian(r):
        return -np.eye(N) + J * (1 - phi(r)**2)
    
    # Use solve_ivp with method 'RK45' or 'LSODA'
    sol = solve_ivp(network_dynamics, [0, t_sim], r0, args=(J,), t_eval=t_eval, method='RK45')
    
    trajectory = sol.y.T
    
    Q = np.eye(N)
    lyap = np.zeros(N)
    
    for i in range(1, len(t_eval)):
        J_t = jacobian(trajectory[i])
        
        try:
            Q_new = expm(J_t * dt) @ Q
        except OverflowError:
            print(f"Overflow encountered in matrix exponential at step {i}, g = {g}.")
            break
        
        # QR decomposition
        Q, R = qr(Q_new)
        lyap += np.log(np.abs(np.diag(R)))
    
    lyap /= t_sim
    return np.sort(lyap)[::-1]

# Simulation parameters
t_sim = 200  # Increased to get smoother results
dt = 0.05    # Decreased to increase resolution
g_values = np.logspace(0, 3, 100)  # g values for both cases

# Define N values for comparison
N_values = [100, 200]

# Initialize lists for storing the results
lyap_spectra_all = {N: [] for N in N_values}
entropy_rates_all = {N: [] for N in N_values}
attractor_dims_all = {N: [] for N in N_values}

# Loop over both N values with tqdm progress bar
for N in N_values:
    print(f"\nCalculating for N = {N}")
    for g in tqdm(g_values, desc=f"g-values for N={N}"):
        lyap = calculate_lyapunov_spectrum(g, N, t_sim, dt)
        lyap_spectra_all[N].append(lyap)
        
        positive_lyap = lyap[lyap > 0]
        entropy_rates_all[N].append(np.sum(positive_lyap))
        
        k = np.argmax(np.cumsum(lyap) < 0)
        attractor_dims_all[N].append(k + np.sum(lyap[:k]) / np.abs(lyap[k]))

# Square plots
plt.figure(figsize=(8, 8))

# Plot (a) Lyapunov spectrum for different g
plt.subplot(221)
g_values_a = [30, 20, 10, 1]
colors = ['red', 'purple', 'blue', 'green']
for g, color in zip(g_values_a, colors):
    lyap = calculate_lyapunov_spectrum(g, 100, t_sim, dt)
    plt.semilogx(range(1, 100+1), lyap, color=color, label=f'g = {g}')
plt.ylim(-5, 5)
plt.xlabel('i')
plt.ylabel('λ (1/τ)')
plt.title('(a) Lyapunov Spectrum')
plt.legend()
plt.grid(True)

# Plot (b) Max Lyapunov exponent λmax vs g
plt.subplot(222)
for N, color in zip([100, 200], ['black', 'orange']):
    plt.semilogx(g_values, [spec[0] for spec in lyap_spectra_all[N]], color=color, label=f'N = {N}')
plt.xlabel('g')
plt.ylabel('λmax (1/τ)')
plt.title('(b) Maximum Lyapunov Exponent')
plt.axhline(y=0, color='k', linestyle='--')
plt.grid(True)
plt.legend()

# Plot (c) Entropy rate vs g
plt.subplot(223)
for N, color in zip([100, 200], ['black', 'orange']):
    smoothed_entropy = np.convolve(entropy_rates_all[N], np.ones(5)/5, mode='valid')  # Smoothing
    plt.semilogx(g_values[:len(smoothed_entropy)], smoothed_entropy, color=color, label=f'N = {N}')
plt.xlabel('g')
plt.ylabel('H/(N/τ)')
plt.title('(c) Entropy Rate')
plt.grid(True)
plt.legend()

# Plot (d) Attractor dimension D/N vs g
plt.subplot(224)
for N, color in zip([100, 200], ['black', 'orange']):
    smoothed_dims = np.convolve([d/N*100 for d in attractor_dims_all[N]], np.ones(5)/5, mode='valid')  # Smoothing
    plt.semilogx(g_values[:len(smoothed_dims)], smoothed_dims, color=color, label=f'N = {N}')
plt.xlabel('g')
plt.ylabel('D/N(%)')
plt.title('(d) Attractor Dimension')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
