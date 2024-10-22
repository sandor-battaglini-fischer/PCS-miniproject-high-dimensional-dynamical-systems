import random
import matplotlib.pyplot as plt

def tent_map(x, mu):
    return mu * x if x <= 0.5 else mu * (1 - x)

def logistic_map(x, mu):
    return mu*x*(1-x)

def plot_bifurcation_diagram(resolution=1000, x_min=0, x_max=2):
    for i in range(resolution):
        mu = x_min + (x_max - x_min) / resolution * i
        x = random.random()
        for t in range(500):
            x = tent_map(x, mu)
            if t > 480:
                plt.plot(mu, x, 'o', c='b', ms=1, alpha=0.5)

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$x$')
    plt.axhline(0.5, ls=':', c='k')
    plt.axvline(1.0, ls=':', c='k')
    plt.xlim(x_min, x_max)
    plt.show()

plot_bifurcation_diagram()
