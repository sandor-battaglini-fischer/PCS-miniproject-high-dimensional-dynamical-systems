import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def wigner_semicircle(x, R):
    return (2 / (np.pi * R**2)) * np.sqrt(R**2 - x**2)

N = 10000  
g_values = [0.9, 1, 1.2] 
num_matrices = 100 

# Plot
plt.figure(figsize=(12, 8))

for g in g_values:
    all_eigenvalues = []
    for _ in range(num_matrices):
        J = np.random.normal(0, (g*g)/np.sqrt(N), (N, N))
        J = (J + J.T) / 2  
        
        eigenvalues = np.linalg.eigvals(J)
        all_eigenvalues.extend(eigenvalues.real)

    # Theoretical distribution
    R = g
    x_theory = np.linspace(-R, R, 1000)
    y_theory = wigner_semicircle(x_theory, R)

    # Histogram
    hist, bin_edges, _ = plt.hist(all_eigenvalues, bins=50, density=True, alpha=0.6, label=f'g={g}')

    # Plot theoretical distribution
    # plt.plot(x_theory, y_theory, linewidth=2)

plt.title(f"Eigenvalue Distribution (N={N})", fontsize=16)
plt.xlabel("Eigenvalue λ", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-max(g_values)-0.5, max(g_values)+0.5)
plt.ylim(0, 0.8) 
# plt.gca().set_aspect('equal', adjustable='box') 

plt.tight_layout()
plt.show()