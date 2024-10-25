import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define Lorenz system
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Generate Lorenz data
def generate_lorenz_data(T, dt=0.01):
    t_span = [0, T]
    t_eval = np.arange(0, T, dt)
    initial_state = [1.0, 1.0, 1.0]
    
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
    return sol.t, sol.y.T

# Parameters from the table
N = 300  # Reservoir size
spectral_radius = 1.4
input_scaling = 0.1
leak_rate = 0.2
T_train = 2000

# Generate training data
t_train, data_train = generate_lorenz_data(T_train)

# Initialize reservoir
np.random.seed(42)
W_in = input_scaling * (np.random.rand(N, 3) * 2 - 1)  # d = 6, system has 3 inputs (x, y, z)
W = np.random.rand(N, N) - 0.5

# Scale reservoir matrix to have a spectral radius < 1
eigenvalues, _ = la.eig(W)
W *= spectral_radius / np.max(np.abs(eigenvalues))

# Train reservoir
reservoir_state = np.zeros(N)
reservoir_states = []
for i in range(len(data_train)):
    u = data_train[i]
    reservoir_state = (1 - leak_rate) * reservoir_state + leak_rate * np.tanh(np.dot(W, reservoir_state) + np.dot(W_in, u))
    reservoir_states.append(np.copy(reservoir_state))

reservoir_states = np.array(reservoir_states)

# Linear regression to find output weights
W_out = la.lstsq(reservoir_states, data_train, cond=None)[0]

# Short-term prediction
T_test = 20  # Testing time
t_test, data_test = generate_lorenz_data(T_test)
predictions = []
reservoir_state = np.copy(reservoir_states[-1])

for i in range(len(data_test)):
    prediction = np.dot(W_out.T, reservoir_state)
    predictions.append(prediction)
    reservoir_state = (1 - leak_rate) * reservoir_state + leak_rate * np.tanh(np.dot(W, reservoir_state) + np.dot(W_in, prediction))

predictions = np.array(predictions)

# Visualization with smaller figure and subtitle
plt.figure(figsize=(8, 6))  # Smaller figure size

plt.suptitle(f'Lorenz Attractor: $D_r={N}$, $T={T_train}$, $\\rho={spectral_radius}$, $\\beta={leak_rate}$, $\\sigma={input_scaling}$', fontsize=14)

plt.subplot(3, 1, 1)
plt.plot(t_test, data_test[:, 0], 'b', label='Actual')
plt.plot(t_test, predictions[:, 0], 'r', label='Predicted')
plt.ylabel('x(t)')
plt.legend(loc='upper right')  # Move legend to top-right corner

plt.subplot(3, 1, 2)
plt.plot(t_test, data_test[:, 1], 'b')
plt.plot(t_test, predictions[:, 1], 'r')
plt.ylabel('y(t)')

plt.subplot(3, 1, 3)
plt.plot(t_test, data_test[:, 2], 'b')
plt.plot(t_test, predictions[:, 2], 'r')
plt.xlabel('Time')
plt.ylabel('z(t)')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit subtitle
plt.show()


# Improvements:

# Increase Reservoir Size: Larger reservoirs can capture more complex dynamics but may require more computational power.
# Tune Hyperparameters: Adjust spectral radius, input scaling, and leak rate.
# Use Regularization: Regularization during linear regression can enhance performance.
# Different Architectures: Investigate echo state networks (ESN)