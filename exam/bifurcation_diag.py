import random
import matplotlib.pyplot as plt

def tent_map(x, mu):
    return mu * x if x <= 0.5 else mu * (1 - x)
resolution = 1000
for i in range(resolution):
    mu = 2/resolution * i
    x = random.random()
    for t in range(500):
        x = tent_map(x, mu)
        if t > 480:
            plt.plot(mu, x, 'o', c = 'b', ms = 1, alpha = 0.5)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$x$')
plt.axhline(0.5, ls=':', c='k')
plt.axvline(1.0, ls=':', c='k')
plt.show()