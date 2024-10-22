import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

def plot_phase_diagram(func, x_range, y_range, params=None, n_points=20):
    """
    Plot a phase diagram for a 2D dynamical system.
    
    Parameters:
    func : callable
        The function describing the dynamical system. Should take arguments (X, t, *params).
    x_range : tuple
        The range of x values to plot.
    y_range : tuple
        The range of y values to plot.
    params : tuple, optional
        Additional parameters for the function.
    n_points : int, optional
        Number of points to use in each dimension.
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    if params is None:
        params = ()
    
    U, V = func([X, Y], 0, *params)
    
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, U, V)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Diagram')
    plt.show()

def plot_bifurcation(func, param_range, x0, n_iterations=1000, n_discard=100, other_params=None):
    """
    Plot a bifurcation diagram for a discrete map.
    
    Parameters:
    func : callable
        The function describing the map. Should take arguments (x, *params).
    param_range : tuple
        The range of parameter values to plot.
    x0 : float or array-like
        Initial condition(s).
    n_iterations : int, optional
        Number of iterations for each parameter value.
    n_discard : int, optional
        Number of initial iterations to discard.
    other_params : tuple, optional
        Other fixed parameters for the function.
    """
    param_vals = np.linspace(param_range[0], param_range[1], 1000)
    x_vals = []

    for param in param_vals:
        x = np.array(x0).flatten()  # Ensure x is a flattened array
        params = (param,) if other_params is None else (param, *other_params)
        for _ in range(n_discard):
            x = np.atleast_1d(func(x, *params)).flatten()
        for _ in range(n_iterations - n_discard):
            x = np.atleast_1d(func(x, *params)).flatten()
            x_vals.extend(zip([param] * len(x), x))

    x_vals = np.array(x_vals)
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_vals[:, 0], x_vals[:, 1], ',k', alpha=0.1, markersize=0.1)
    plt.xlabel('Parameter')
    plt.ylabel('x')
    plt.title('Bifurcation Diagram')
    plt.show()

def calculate_entropy(sequence, base=2):
    """
    Calculate the Shannon entropy of a sequence.
    
    Parameters:
    sequence : array-like
        The sequence of symbols.
    base : float, optional
        The base of the logarithm to use.
    
    Returns:
    float
        The calculated entropy.
    """
    _, counts = np.unique(sequence, return_counts=True)
    probs = counts / len(sequence)
    return -np.sum(probs * np.log(probs) / np.log(base))

def calculate_lyapunov(func, x0, n_iterations=1000, epsilon=1e-6, params=None):
    """
    Calculate the Lyapunov exponent for a map.
    
    Parameters:
    func : callable
        The function describing the map. Should take arguments (x, *params).
    x0 : float or array-like
        Initial condition(s).
    n_iterations : int, optional
        Number of iterations.
    epsilon : float, optional
        Small perturbation for finite difference approximation.
    params : tuple, optional
        Parameters for the function.
    
    Returns:
    float or array
        The calculated Lyapunov exponent(s).
    """
    if params is None:
        params = ()
    
    x = np.array(x0)
    lyap = np.zeros_like(x)
    for _ in range(n_iterations):
        x_perturbed = x + epsilon
        dx = func(x_perturbed, *params) - func(x, *params)
        lyap += np.log(np.abs(dx / epsilon))
        x = func(x, *params)
    
    return lyap / n_iterations

def plot_time_series(func, x0, n_iterations, params=None, discrete=True):
    """
    Plot the time series for a dynamical system.
    
    Parameters:
    func : callable
        The function describing the system. For discrete systems, should take arguments (x, *params).
        For continuous systems, should take arguments (X, t, *params).
    x0 : float or array-like
        Initial condition(s).
    n_iterations : int
        Number of iterations or time points.
    params : tuple, optional
        Additional parameters for the function.
    discrete : bool, optional
        Whether the system is discrete (True) or continuous (False).
    """
    if params is None:
        params = ()
    
    if discrete:
        x = np.zeros(n_iterations)
        x[0] = x0
        for i in range(1, n_iterations):
            x[i] = func(x[i-1], *params)
        t = np.arange(n_iterations)
    else:
        t = np.linspace(0, n_iterations, n_iterations)
        x = odeint(func, x0, t, args=params)
    
    plt.figure(figsize=(10, 6))
    if x.ndim == 1:
        plt.plot(t, x)
        plt.ylabel('x')
    else:
        for i in range(x.shape[1]):
            plt.plot(t, x[:, i], label=f'x{i+1}')
        plt.legend()
    plt.xlabel('t')
    plt.title('Time Series')
    plt.show()

def poincare_section(func, x0, n_iterations, plane_coord, plane_value, params=None):
    """
    Generate a Poincaré section for a continuous dynamical system using odeint.
    
    Parameters:
    func : callable
        The function describing the system. Should take arguments (X, t, *params).
    x0 : array-like
        Initial conditions.
    n_iterations : int
        Number of time points to simulate.
    plane_coord : int
        The coordinate of the plane (0, 1, or 2 for x, y, or z).
    plane_value : float
        The value of the plane.
    params : tuple, optional
        Additional parameters for the function.
    
    Returns:
    array
        The Poincaré section points.
    """
    if params is None:
        params = ()
    
    t = np.linspace(0, 100, n_iterations)
    trajectory = odeint(func, x0, t, args=params)
    
    # Find points where the trajectory crosses the plane
    signs = np.sign(trajectory[:, plane_coord] - plane_value)
    indices = np.where(np.diff(signs))[0]
    
    print(f"Number of plane crossings: {len(indices)}")
    
    # Linear interpolation to find the exact crossing points
    section = []
    for i in indices:
        t1, t2 = t[i], t[i+1]
        y1, y2 = trajectory[i], trajectory[i+1]
        
        # Interpolation factor
        f = (plane_value - y1[plane_coord]) / (y2[plane_coord] - y1[plane_coord])
        
        # Interpolated point
        point = y1 + f * (y2 - y1)
        section.append(point[[j for j in range(len(x0)) if j != plane_coord]])
    
    return np.array(section)

def plot_poincare_section(section):
    """
    Plot a Poincaré section.
    
    Parameters:
    section : array
        The Poincaré section points.
    """
    plt.figure(figsize=(8, 8))
    if section.size == 0:
        print("Error: Empty Poincaré section")
        return
    
    if section.ndim == 1:
        plt.plot(section, np.zeros_like(section), '.k')
        plt.xlabel('x')
        plt.ylabel('y')
    elif section.ndim == 2:
        if section.shape[1] == 2:
            plt.plot(section[:, 0], section[:, 1], '.k')
            plt.xlabel('x')
            plt.ylabel('y')
        elif section.shape[1] == 3:
            ax = plt.gca(projection='3d')
            ax.plot(section[:, 0], section[:, 1], section[:, 2], '.k')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            print(f"Error: Unexpected shape of Poincaré section: {section.shape}")
            return
    else:
        print(f"Error: Unexpected dimensions of Poincaré section: {section.ndim}")
        return
    
    plt.title('Poincaré Section')
    plt.show()
    
def plot_3d_system(func, x0, t_span, params=None):
    """
    Create a 3D plot of a dynamical system.
    
    Parameters:
    func : callable
        The function describing the system. Should take arguments (X, t, *params).
    x0 : array-like
        Initial conditions.
    t_span : tuple
        The time span for integration (t_start, t_end).
    params : tuple, optional
        Additional parameters for the function.
    """
    if params is None:
        params = ()
    
    t = np.linspace(t_span[0], t_span[1], 10000)
    solution = odeint(func, x0, t, args=params)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Dynamical System')
    
    plt.show()

def calculate_fractal_dimension(attractor, eps_range):
    """
    Calculate the fractal dimension of an attractor using the box-counting method.
    
    Parameters:
    attractor : array
        The points of the attractor.
    eps_range : array
        The range of box sizes to use.
    
    Returns:
    float
        The estimated fractal dimension.
    """
    N = []
    for eps in eps_range:
        scaled = np.floor(attractor / eps)
        N.append(len(np.unique(scaled, axis=0)))
    
    coeffs = np.polyfit(np.log(1/eps_range), np.log(N), 1)
    return coeffs[0]

def plot_cobweb(func, x0, n_iterations, *params, x_range=(0, 1)):
    """
    Plot the cobweb diagram for a given map.

    Parameters:
    func : callable
        The function describing the map. Should take arguments (x, *params).
    x0 : float
        Initial condition.
    n_iterations : int
        Number of iterations to plot.
    *params : float
        Parameters for the map function.
    x_range : tuple, optional
        Range of x values to plot.

    Returns:
    None
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = np.array([func(xi, *params) for xi in x])

    plt.figure(figsize=(10, 10))
    plt.plot(x, y, 'k')
    plt.plot(x, x, 'k')

    xi = x0
    for _ in range(n_iterations):
        yi = func(xi, *params)
        plt.plot([xi, xi], [xi, yi], 'r', linewidth=0.5)
        plt.plot([xi, yi], [yi, yi], 'r', linewidth=0.5)
        xi = yi

    plt.xlabel('x_n')
    plt.ylabel('x_{n+1}')
    plt.title('Cobweb Diagram')
    plt.axis('equal')
    plt.show()

def calculate_correlation_dimension(attractor, r_range, max_pairs=10000):
    """
    Calculate the correlation dimension of an attractor.
    
    Parameters:
    attractor : array
        The points of the attractor.
    r_range : array
        The range of radii to use.
    max_pairs : int, optional
        Maximum number of point pairs to use for calculation.
    
    Returns:
    float
        The estimated correlation dimension.
    """
    N = len(attractor)
    if N * (N - 1) / 2 > max_pairs:
        indices = np.random.choice(N, size=int(np.sqrt(2 * max_pairs)), replace=False)
        attractor = attractor[indices]
        N = len(attractor)
    
    distances = np.sqrt(((attractor[:, None, :] - attractor[None, :, :]) ** 2).sum(axis=2))
    distances = distances[np.triu_indices(N, k=1)]
    
    C = np.array([(distances <= r).sum() / (N * (N - 1) / 2) for r in r_range])
    
    # Filter out zero values and corresponding r values
    valid_indices = C > 0
    valid_C = C[valid_indices]
    valid_r = r_range[valid_indices]
    
    if len(valid_C) < 2:
        print("Error: Not enough valid points to calculate correlation dimension")
        return np.nan
    
    coeffs = np.polyfit(np.log(valid_r), np.log(valid_C), 1)
    return coeffs[0]

def plot_recurrence(time_series, threshold, delay=1, embed_dim=1):
    """
    Create a recurrence plot for a time series.
    
    Parameters:
    time_series : array
        The input time series.
    threshold : float
        The threshold distance for considering points as recurrent.
    delay : int, optional
        The delay for time-delay embedding.
    embed_dim : int, optional
        The embedding dimension.
    """
    N = len(time_series)
    if embed_dim > 1:
        embedded = np.array([time_series[i:i+embed_dim*delay:delay] for i in range(N - (embed_dim-1)*delay)])
    else:
        embedded = time_series.reshape(-1, 1)
    
    distances = np.sqrt(((embedded[:, None, :] - embedded[None, :, :]) ** 2).sum(axis=2))
    recurrence = distances < threshold
    
    plt.figure(figsize=(8, 8))
    plt.imshow(recurrence, cmap='binary', origin='lower')
    plt.colorbar(label='Recurrence')
    plt.xlabel('Time')
    plt.ylabel('Time')
    plt.title('Recurrence Plot')
    plt.show()

def calculate_kld(p, q):
    """
    Calculate the Kullback-Leibler divergence between two distributions.
    
    Parameters:
    p : array
        The first probability distribution.
    q : array
        The second probability distribution.
    
    Returns:
    float
        The KL divergence.
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def calculate_mutual_information(x, y, bins=10):
    """
    Calculate the mutual information between two variables.
    
    Parameters:
    x : array
        The first variable.
    y : array
        The second variable.
    bins : int, optional
        The number of bins to use for discretization.
    
    Returns:
    float
        The calculated mutual information.
    """
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    hist_x, _ = np.histogram(x, bins=bins)
    hist_y, _ = np.histogram(y, bins=bins)
    
    p_xy = hist_xy / np.sum(hist_xy)
    p_x = hist_x / np.sum(hist_x)
    p_y = hist_y / np.sum(hist_y)
    
    p_x_y = p_x[:, np.newaxis] * p_y[np.newaxis, :]
    
    mi = np.sum(p_xy * np.log2(p_xy / p_x_y + 1e-10))
    return mi

def false_nearest_neighbors(time_series, max_dim, delay=1, R=10, A=2):
    """
    Perform the false nearest neighbors test to estimate embedding dimension.
    
    Parameters:
    time_series : array
        The input time series.
    max_dim : int
        The maximum embedding dimension to test.
    delay : int, optional
        The delay for time-delay embedding.
    R : float, optional
        The threshold for considering a neighbor as false.
    A : float, optional
        The threshold for the relative increase in distance.
    
    Returns:
    array
        The fraction of false nearest neighbors for each dimension.
    """
    N = len(time_series)
    fnn_fractions = []
    
    for dim in range(1, max_dim + 1):
        embedded = np.array([time_series[i:i+dim*delay:delay] for i in range(N - (dim-1)*delay)])
        
        distances = np.linalg.norm(embedded[:, None, :] - embedded[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        
        nearest_indices = np.argmin(distances, axis=1)
        
        if dim < max_dim:
            next_embedded = np.array([time_series[i:i+(dim+1)*delay:delay] for i in range(N - dim*delay)])
            next_distances = np.abs(next_embedded[:, -1] - next_embedded[nearest_indices[:-delay], -1])
            
            fnn = np.sum((next_distances / distances[:-delay, nearest_indices[:-delay]] > R) | 
                         (next_distances / np.std(time_series) > A))
            fnn_fractions.append(fnn / (N - dim*delay))
        else:
            fnn_fractions.append(0)
    
    return np.array(fnn_fractions)

def plot_fnn(fnn_fractions):
    """
    Plot the results of the false nearest neighbors test.
    
    Parameters:
    fnn_fractions : array
        The fraction of false nearest neighbors for each dimension.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fnn_fractions) + 1), fnn_fractions, 'bo-')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Fraction of False Nearest Neighbors')
    plt.title('False Nearest Neighbors Test')
    plt.show()


def numerical_jacobian(func, x, t, params, eps=1e-8):
    """Calculate the Jacobian matrix using finite differences."""
    n = len(x)
    J = np.zeros((n, n))
    f0 = np.array(func(x, t, *params))
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += eps
        f1 = np.array(func(x_perturbed, t, *params))
        J[:, i] = (f1 - f0) / eps
    return J


def calculate_lyapunov_spectrum(func, x0, t_final, dt, dim, params=()):
    """
    Calculate the Lyapunov spectrum for a continuous dynamical system.
    
    Parameters:
    func : callable
        The function describing the dynamical system. Should take arguments (x, t, *params).
    x0 : array-like
        Initial condition.
    t_final : float
        Final time for integration.
    dt : float
        Time step for integration.
    dim : int
        Dimension of the system.
    params : tuple, optional
        Additional parameters for the function.
    
    Returns:
    array
        The Lyapunov spectrum.
    """
    n_steps = int(t_final / dt)
    t = np.linspace(0, t_final, n_steps)
    
    def system(X, t):
        x, Q = X[:dim], X[dim:].reshape(dim, dim)
        dx = func(x, t, *params)
        jac = numerical_jacobian(func, x, t, params)
        dQ = np.dot(jac, Q)
        return np.concatenate([dx, dQ.flatten()])
    
    X0 = np.concatenate([x0, np.eye(dim).flatten()])
    
    # Use a more robust ODE solver
    solution = odeint(system, X0, t, rtol=1e-8, atol=1e-8)
    
    Q = solution[:, dim:].reshape(-1, dim, dim)
    R = np.zeros((n_steps, dim))
    
    lyap = np.zeros(dim)
    for i in range(n_steps):
        Q[i], R_step = np.linalg.qr(Q[i])
        R[i] = np.abs(np.diagonal(R_step))
        lyap += np.log(R[i] + 1e-16)  # Add small constant to avoid log(0)
    
    lyap /= t_final
    return lyap

def plot_lyapunov_spectrum(lyap_spectrum):
    """
    Plot the Lyapunov spectrum.
    
    Parameters:
    lyap_spectrum : array
        The Lyapunov spectrum.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lyap_spectrum) + 1), lyap_spectrum, 'bo-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Lyapunov Exponent Index')
    plt.ylabel('Lyapunov Exponent')
    plt.title('Lyapunov Spectrum')
    plt.show()

def calculate_kolmogorov_sinai_entropy(lyapunov_spectrum):
    """
    Calculate the Kolmogorov-Sinai entropy from the Lyapunov spectrum.
    
    Parameters:
    lyapunov_spectrum : array-like
        The Lyapunov spectrum.
    
    Returns:
    float
        The Kolmogorov-Sinai entropy.
    """
    positive_exponents = [exponent for exponent in lyapunov_spectrum if exponent > 0]
    return sum(positive_exponents)

def plot_bifurcation_2d(func, param1_range, param2_range, x0, n_iterations=1000, n_discard=100):
    """
    Plot a 2D bifurcation diagram.
    
    Parameters:
    func : callable
        The function describing the map. Should take arguments (x, r1, r2).
    param1_range : tuple
        The range of the first parameter values to plot.
    param2_range : tuple
        The range of the second parameter values to plot.
    x0 : float
        Initial condition.
    n_iterations : int, optional
        Number of iterations for each parameter value.
    n_discard : int, optional
        Number of initial iterations to discard.
    """
    r1_vals = np.linspace(param1_range[0], param1_range[1], 100)
    r2_vals = np.linspace(param2_range[0], param2_range[1], 100)
    
    bifurcation_data = np.zeros((len(r1_vals), len(r2_vals)))
    
    for i, r1 in enumerate(r1_vals):
        for j, r2 in enumerate(r2_vals):
            x = x0
            for _ in range(n_discard):
                x = func(x, r1, r2)
            for _ in range(n_iterations - n_discard):
                x = func(x, r1, r2)
                bifurcation_data[i, j] += x
    
    bifurcation_data /= (n_iterations - n_discard)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(bifurcation_data.T, extent=[param1_range[0], param1_range[1], param2_range[0], param2_range[1]], 
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Average x value')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('2D Bifurcation Diagram')
    plt.show()

def calculate_information_dimension(attractor, r_range):
    """
    Calculate the information dimension of an attractor.
    
    Parameters:
    attractor : array
        The points of the attractor.
    r_range : array
        The range of radii to use.
    
    Returns:
    float
        The estimated information dimension.
    """
    N = len(attractor)
    distances = np.sqrt(((attractor[:, None, :] - attractor[None, :, :]) ** 2).sum(axis=2))
    
    I = []
    for r in r_range:
        p = (distances < r).sum(axis=1) / N
        I.append(-np.sum(p * np.log(p + 1e-10)) / N)
    
    coeffs = np.polyfit(np.log(r_range), I, 1)
    return coeffs[0]

def plot_power_spectrum(time_series, dt):
    """
    Plot the power spectrum of a time series.
    
    Parameters:
    time_series : array
        The input time series.
    dt : float
        The time step between consecutive points.
    """
    N = len(time_series)
    freqs = np.fft.fftfreq(N, dt)
    ps = np.abs(np.fft.fft(time_series))**2
    
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs[1:N//2], ps[1:N//2])
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
    plt.show()

def calculate_correlation_sum(attractor, r_range, max_pairs=10000):
    """
    Calculate the correlation sum for an attractor.
    
    Parameters:
    attractor : array
        The points of the attractor.
    r_range : array
        The range of radii to use.
    max_pairs : int, optional
        Maximum number of point pairs to use for calculation.
    
    Returns:
    array
        The correlation sum for each radius.
    """
    N = len(attractor)
    if N * (N - 1) / 2 > max_pairs:
        indices = np.random.choice(N, size=int(np.sqrt(2 * max_pairs)), replace=False)
        attractor = attractor[indices]
        N = len(attractor)
    
    distances = np.sqrt(((attractor[:, None, :] - attractor[None, :, :]) ** 2).sum(axis=2))
    distances = distances[np.triu_indices(N, k=1)]
    
    return np.array([(distances <= r).sum() / (N * (N - 1) / 2) for r in r_range])

def plot_correlation_sum(r_range, C):
    """
    Plot the correlation sum.
    
    Parameters:
    r_range : array
        The range of radii used.
    C : array
        The correlation sum for each radius.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(r_range, C, 'bo-')
    plt.xlabel('r')
    plt.ylabel('C(r)')
    plt.title('Correlation Sum')
    plt.show()

def calculate_generalized_dimensions(attractor, q_range, r_range, max_pairs=10000):
    """
    Calculate the generalized dimensions (Renyi dimensions) of an attractor.
    
    Parameters:
    attractor : array
        The points of the attractor.
    q_range : array
        The range of q values to use.
    r_range : array
        The range of radii to use.
    max_pairs : int, optional
        Maximum number of point pairs to use for calculation.
    
    Returns:
    array
        The generalized dimensions for each q value.
    """
    N = len(attractor)
    if N * (N - 1) / 2 > max_pairs:
        indices = np.random.choice(N, size=int(np.sqrt(2 * max_pairs)), replace=False)
        attractor = attractor[indices]
        N = len(attractor)
    
    distances = np.sqrt(((attractor[:, None, :] - attractor[None, :, :]) ** 2).sum(axis=2))
    
    D_q = []
    for q in q_range:
        if q == 1:
            I = []
            for r in r_range:
                p = (distances < r).sum(axis=1) / N
                I.append(-np.sum(p * np.log(p + 1e-10)) / N)
            coeffs = np.polyfit(np.log(r_range), I, 1)
            D_q.append(coeffs[0])
        else:
            Y = []
            for r in r_range:
                Y.append(np.log(np.sum(((distances < r).sum(axis=1) / N) ** (q - 1))))
            coeffs = np.polyfit(np.log(r_range), Y, 1)
            D_q.append(coeffs[0] / (q - 1))
    
    return np.array(D_q)

def plot_generalized_dimensions(q_range, D_q):
    """
    Plot the generalized dimensions.
    
    Parameters:
    q_range : array
        The range of q values used.
    D_q : array
        The generalized dimensions for each q value.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(q_range, D_q, 'bo-')
    plt.xlabel('q')
    plt.ylabel('D_q')
    plt.title('Generalized Dimensions')
    plt.show()

def calculate_sample_entropy(time_series, m, r):
    """
    Calculate the sample entropy of a time series.
    
    Parameters:
    time_series : array
        The input time series.
    m : int
        Embedding dimension.
    r : float
        Tolerance (typically 0.1 to 0.25 times the standard deviation of the time series).
    
    Returns:
    float
        The sample entropy.
    """
    N = len(time_series)
    tempX = np.array([time_series[i:i+m] for i in range(N-m+1)])
    tempX1 = np.array([time_series[i:i+m+1] for i in range(N-m)])
    
    B = np.sum([np.sum(np.max(np.abs(tempX[i] - tempX), axis=1) <= r) - 1 for i in range(N-m+1)])
    A = np.sum([np.sum(np.max(np.abs(tempX1[i] - tempX1), axis=1) <= r) - 1 for i in range(N-m)])
    
    return -np.log(A / B)

def recurrence_quantification_analysis(recurrence_matrix):
    """
    Perform recurrence quantification analysis on a recurrence matrix.
    
    Parameters:
    recurrence_matrix : array
        The recurrence matrix.
    
    Returns:
    dict
        A dictionary containing RQA measures.
    """
    N = recurrence_matrix.shape[0]
    
    # Recurrence Rate
    RR = np.sum(recurrence_matrix) / (N * N)
    
    # Determinism
    diag_lengths = []
    for i in range(-(N-1), N):
        diag = np.diagonal(recurrence_matrix, offset=i)
        diag_lengths.extend(np.diff(np.where(np.concatenate(([diag[0]], diag[:-1] != diag[1:], [True])))[0])[diag])
    
    DET = np.sum([l for l in diag_lengths if l >= 2]) / np.sum(diag_lengths)
    
    # Average Diagonal Line Length
    L = np.mean(diag_lengths) if diag_lengths else 0
    
    # Laminarity
    vert_lengths = []
    for col in recurrence_matrix.T:
        vert_lengths.extend(np.diff(np.where(np.concatenate(([col[0]], col[:-1] != col[1:], [True])))[0])[col])
    
    LAM = np.sum([l for l in vert_lengths if l >= 2]) / np.sum(vert_lengths)
    
    return {'RR': RR, 'DET': DET, 'L': L, 'LAM': LAM}














