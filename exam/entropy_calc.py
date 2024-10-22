import numpy as np

def tent_map(x, mu):
    return mu * x if x <= 0.5 else mu * (1 - x)

mu = 1.99
N = 5000


sigma = np.zeros(N)
x = np.random.uniform()
for t in range(N):
    x = tent_map(x, mu)
    sigma[t] = (x > 0.5)
p = (sigma @ sigma)/N
S = -p * np.log(p) - (1-p) * np.log(1-p)
print(S)
S_log = np.exp(S)
print(S_log)