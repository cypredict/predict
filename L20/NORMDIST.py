from scipy import stats
mu = 179.5
sigma = 3.697
x = 180
prob = stats.norm.pdf(x, mu, sigma)
print(prob)

import numpy as np
print(np.log2(0.4))
print(np.log2(0.2))