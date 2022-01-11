import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt

shape = 1.1039801030723038
loc = 0.0037469909687539286
scale = 0.0022326553170845956
fig, ax = plt.subplots(1, 1)
c = 0.1
x = np.linspace(loc, 0.1, 100)

ax.plot(x, genpareto.pdf(x, shape, loc=loc, scale=scale), 'r-', lw=5, alpha=0.6, label='genpareto pdf')
plt.show()