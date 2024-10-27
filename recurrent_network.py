import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ReservoirComputer:
    def __init__(self, N, spectral_radius, input_scaling, leak_rate, seed=42):
        self.N = N  # Reservoir size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        
        # Initialize weights
        np.random.seed(seed)
        self.W_in = input_scaling * (np.random.rand(N, 3) * 2 - 1)
        self.W = np.random.rand(N, N) - 0.5
        
        # Scale reservoir matrix
        eigenvalues, _ = la.eig(self.W)
        self.W *= spectral_radius / np.max(np.abs(eigenvalues))
        
        self.W_out = None  # Will be set during training
        
    def train(self, data):
        """Train the reservoir using the provided data."""
        reservoir_state = np.zeros(self.N)
        reservoir_states = []
        
        # Run reservoir with training data
        for u in data:
            reservoir_state = self._update_state(reservoir_state, u)
            reservoir_states.append(np.copy(reservoir_state))
            
        reservoir_states = np.array(reservoir_states)
        
        # Compute output weights using linear regression
        self.W_out = la.lstsq(reservoir_states, data, cond=None)[0]
        return reservoir_states
    
    def predict(self, initial_state, n_steps):
        """Generate predictions starting from initial_state."""
        predictions = [initial_state]  # Force first prediction to be initial state
        
        # Initialize reservoir state to produce the initial state
        r = np.dot(la.pinv(self.W_out.T), initial_state)
        
        # Generate predictions
        for _ in range(n_steps - 1):
            prediction = np.dot(self.W_out.T, r)
            r = self._update_state(r, prediction)
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def _update_state(self, state, input_data):
        """Update reservoir state using leaky integration."""
        return ((1 - self.leak_rate) * state + 
                self.leak_rate * np.tanh(np.dot(self.W, state) + 
                                       np.dot(self.W_in, input_data)))

def find_maxima(data):
    maxima = []
    for i in range(1, len(data)-1):
        if data[i-1] < data[i] and data[i] > data[i+1]:
            maxima.append(data[i])
    return np.array(maxima)

# Define Lorenz system
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Generate Lorenz data
def generate_lorenz_data(T, dt=0.01, initial_state=None):
    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]
    
    t_span = [0, T]
    t_eval = np.arange(0, T, dt)
    
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
    return sol.t, sol.y.T

def plot_comparison(t, actual, predicted, title):
    """Plot comparison between actual and predicted trajectories."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)
    
    # Find index corresponding to t=25
    t_max = 35
    idx_max = np.where(t >= t_max)[0][0] if len(np.where(t >= t_max)[0]) > 0 else len(t)
    
    labels = ['x', 'y', 'z']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t[:idx_max], actual[:idx_max, i], 'b', label='Actual', linewidth=1)
        ax.plot(t[:idx_max], predicted[:idx_max, i], 'r', label='Predicted', linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right')
    
    # Set x-axis limits
    for ax in axes:
        ax.set_xlim(0, t_max)
    
    axes[-1].set_xlabel('t')
    plt.tight_layout()
    plt.show()
def plot_return_map(actual, predicted):
    """Plot return map of z-coordinate maxima."""
    true_maxima = find_maxima(actual[:, 2])
    pred_maxima = find_maxima(predicted[:, 2])
    
    plt.figure(figsize=(6, 6))
    plt.title('Return Map of $z$-coordinate')
    plt.plot(true_maxima[:-1], true_maxima[1:], 'b.', 
             markersize=5, label='Actual', alpha=0.5)
    plt.plot(pred_maxima[:-1], pred_maxima[1:], 'r.', 
             markersize=5, label='Predicted', alpha=0.5)
    plt.xlabel('$z_n$')
    plt.ylabel('$z_{n+1}$')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.xlim(min(true_maxima), max(true_maxima))
    plt.ylim(min(true_maxima), max(true_maxima))
    
    plt.show()

def calculate_lyapunov_exponent(trajectory, dt, max_steps=None):
    """Calculate the largest Lyapunov exponent using the trajectory."""
    if max_steps is None:
        max_steps = len(trajectory)
    
    # Use only the first max_steps
    trajectory = trajectory[:max_steps]
    
    # Parameters for the algorithm
    delay = 20  # Delay for finding nearest neighbors
    evolve_time = 20  # Number of steps to evolve before calculating divergence
    
    n_steps = len(trajectory)
    lyap = 0.0
    n_exponents = 0
    
    for i in range(delay, n_steps - evolve_time):
        # Find nearest neighbor (excluding points too close in time)
        distances = np.linalg.norm(trajectory[i] - trajectory[delay:i-delay], axis=1)
        if len(distances) == 0:
            continue
        nn_idx = np.argmin(distances) + delay
        
        # Initial distance
        d0 = np.linalg.norm(trajectory[i] - trajectory[nn_idx])
        if d0 == 0:
            continue
            
        # Evolved distance
        d1 = np.linalg.norm(trajectory[i + evolve_time] - trajectory[nn_idx + evolve_time])
        if d1 == 0:
            continue
            
        # Add to the running average
        lyap += np.log(d1/d0) / (evolve_time * dt)
        n_exponents += 1
    
    if n_exponents == 0:
        return None
    
    return lyap / n_exponents

def lyapunov_calculation(t, predicted, window_size=1000):
    """Plot running estimate of largest Lyapunov exponent for both trajectories."""
    dt = t[1] - t[0]
    
    lyap_estimates_pred = []
    

    step_size = window_size 
    
    max_steps = min(len(t), 20000)  # Limit total steps processed
    
    for i in range(window_size, max_steps, step_size):
        le_pred = calculate_lyapunov_exponent(predicted[:i], dt, max_steps=window_size)
        
        if le_pred is not None:
            lyap_estimates_pred.append(le_pred)
    
    print(f"The predicted maximum lyapunov exponent is {max(lyap_estimates_pred)}")


def main():
    # Parameters
    N = 300
    spectral_radius = 1.4
    input_scaling = 0.1
    leak_rate = 0.2
    dt = 0.01
    
    # Create and train reservoir
    rc = ReservoirComputer(N, spectral_radius, input_scaling, leak_rate)
    
    # Generate training data
    T_train = 2000
    _, train_data = generate_lorenz_data(T_train)
    rc.train(train_data)
    
    # Generate predictions
    T_pred = 2000
    initial_state = [1.0, 1.0, 1.0]
    t_pred, actual_data = generate_lorenz_data(T_pred, initial_state=initial_state)
    predicted_data = rc.predict(initial_state, len(t_pred))
    
    # Plot results
    plot_comparison(t_pred, actual_data, predicted_data, 
                   f'$D_r={N}$, $\\rho={spectral_radius}$, $\\sigma={input_scaling}$, $\\beta={leak_rate}$')
    # plot_return_map(actual_data, predicted_data)
    lyapunov_calculation(t_pred, predicted_data)

if __name__ == "__main__":
    main()




