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

# Parameters
N = 100  # Reservoir size
spectral_radius = 1.1
input_scaling = 0.1
leak_rate = 0.3

# Generate training data
T_train = 2000
t_train, data_train = generate_lorenz_data(T_train)

# Initialize reservoir
np.random.seed(42)
W_in = input_scaling * (np.random.rand(N, 3) * 2 - 1)
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
T_test = 20
t_test, data_test = generate_lorenz_data(T_test)
predictions = []
reservoir_state = np.copy(reservoir_states[-1])

for i in range(len(data_test)):
    prediction = np.dot(W_out.T, reservoir_state)
    predictions.append(prediction)
    reservoir_state = (1 - leak_rate) * reservoir_state + leak_rate * np.tanh(np.dot(W, reservoir_state) + np.dot(W_in, prediction))

predictions = np.array(predictions)

# Visualization
plt.figure(figsize=(15, 5))
plt.plot(t_test, data_test[:, 0], 'b', label='True x(t)')
plt.plot(t_test, predictions[:, 0], 'r', label='Predicted x(t)', linestyle='--')
plt.title('Lorenz Attractor Prediction')
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.show()


# Improvements:

# Increase Reservoir Size: Larger reservoirs can capture more complex dynamics but may require more computational power.
# Tune Hyperparameters: Adjust spectral radius, input scaling, and leak rate.
# Use Regularization: Regularization during linear regression can enhance performance.
# Different Architectures: Investigate echo state networks (ESN)