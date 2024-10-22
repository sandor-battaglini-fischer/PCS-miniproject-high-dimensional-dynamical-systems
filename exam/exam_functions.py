def tent_map(x, mu):
    return mu * x if x <= 0.5 else mu * (1 - x)

def logistic_map(x, r):
    """Logistic map function"""
    return r * x * (1 - x)

def henon_map(x, a, b):
    return 1 - a * x**2 + b * x

def lorenz_system(X, t, sigma=10, rho=28, beta=8/3):
    """Lorenz system equations"""
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def rossler_system(X, t, a=0.2, b=0.2, c=5.7):
    x, y, z = X
    dx = -y - z
    dy = x + a*y
    dz = b + z*(x - c)
    return [dx, dy, dz]