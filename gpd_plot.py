import matplotlib.pyplot as plt
import numpy as np

xi_plus = np.array([0.1, 0.5, 1, 2])
xi_minus = np.array([-1.1, -1, -0.8, -0.5, -0.3])
sigma = 1
mu = 0

x_plus = np.linspace(0, 8, 1000)
x_minus = np.linspace(0, 3, 1000)
line_style = ["solid", "dotted", "dashed", 'dashdot', (0, (3, 1, 1, 1, 1, 1))]

plt.figure(figsize=(9, 5))

plt.subplot(121)

for idx, exponent in enumerate(xi_plus):
    print(idx, exponent)
    pdf = 1 / sigma * (1 + exponent * (x_plus - mu) / sigma) ** (-1 * (1 / exponent + 1))
    plt.plot(x_plus, np.nan_to_num(pdf), linestyle=line_style[idx], label=r"$\xi = $" + str(exponent))

plt.xlabel(r"$x$")
plt.ylabel(r"$f(x|\xi,\sigma,\mu), \sigma=1, \mu=0$")
plt.legend()


plt.subplot(122)
for idx, exponent in enumerate(xi_minus):
    print(idx, exponent)
    pdf = 1 / sigma * (1 + exponent * (x_minus - mu) / sigma) ** (-1 * (1 / exponent + 1))
    pdf[np.where(x_minus > -1/exponent)] = 0
    plt.plot(x_minus, np.nan_to_num(pdf), linestyle=line_style[idx], label=r"$\xi = $" + str(exponent))

plt.xlabel(r"$x$")
plt.ylabel(r"$f(x|\xi,\sigma,\mu), \sigma=1, \mu=0$")
plt.legend()
plt.tight_layout()
plt.savefig('pdfs.eps', format='eps', dpi=300)
plt.show()

