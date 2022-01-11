import numpy as np
from matplotlib import pyplot as plt

mouse0 = np.genfromtxt("mouse0.csv")
mouse0_elbnd = np.genfromtxt("mouse0_elbnd.csv")
mouse0_le = np.genfromtxt("mouse0_le.csv")

mouse8 = np.genfromtxt("mouse8.csv")
mouse8_elbnd = np.genfromtxt("mouse8_elbnd.csv")
mouse8_le = np.genfromtxt("mouse8_le.csv")

mouse14 = np.genfromtxt("mouse14.csv")
mouse14_elbnd = np.genfromtxt("mouse14_elbnd.csv")
mouse14_le = np.genfromtxt("mouse14_le.csv")

fig, axs = plt.subplots(3, 1, sharex=True)
# fig.subplots_adjust(hspace=0)
print(np.argmax(mouse0))
print(np.argmax(mouse8))
print(np.argmax(mouse14))


axs[0].plot(mouse0)
axs[0].annotate('seizure', xy=(len(mouse0[:])-625, max(mouse0[:])-2), xytext=(1200, max(mouse0[:])-5), arrowprops=dict(facecolor='black', shrink=1))
axs[0].axvline(x=len(mouse0[:])-625, color='k', linestyle=":")
axs[0].set_ylabel('$ESE$ $[-]$')
axs[0].set_title('C3', rotation='horizontal', x=1.04, y=0.4)
axs[1].plot(mouse8)
axs[1].annotate('seizure', xy=(len(mouse8[:])-625, max(mouse8[:])-0.3), xytext=(1200, max(mouse8[:])-0.7), arrowprops=dict(facecolor='black', shrink=1))
axs[1].axvline(x=len(mouse8[:])-625, color='k', linestyle=":")
axs[1].set_ylabel('$ESE$ $[-]$')
axs[1].set_title('Pz', rotation='horizontal', x=1.04, y=0.4)
axs[2].plot(mouse14)
axs[2].annotate('seizure', xy=(len(mouse14[:])-625, max(mouse14[:])-0.3), xytext=(1200, max(mouse14[:])-0.7), arrowprops=dict(facecolor='black', shrink=1))
axs[2].axvline(x=len(mouse14[:])-625, color='k', linestyle=":")
axs[2].set_ylabel('$ESE$ $[-]$')
axs[2].set_title('Fp1', rotation='horizontal', x=1.04, y=0.4)
plt.xlabel('$k$ $[-]$')

plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig('mouse_novelty.eps', format='eps', dpi=300)

fig, axs = plt.subplots(3, 1, sharex=True)
# fig.subplots_adjust(hspace=0)

axs[0].plot(mouse0_elbnd[999:])
axs[0].annotate('seizure', xy=(len(mouse0_elbnd[999:])-625, max(mouse0_elbnd[999:])), xytext=(1200, max(mouse0_elbnd[999:])), arrowprops=dict(facecolor='black', shrink=1))
axs[0].axvline(x=len(mouse0_elbnd[999:])-625, color='k', linestyle=":")
axs[0].set_ylabel('$ELBND$ $[-]$')
axs[0].set_title('C3', rotation='horizontal', x=1.04, y=0.4)
axs[1].plot(mouse8_elbnd[999:])
axs[1].annotate('seizure', xy=(len(mouse8_elbnd[999:])-625, max(mouse8_elbnd[999:])), xytext=(1200, max(mouse8_elbnd[999:])), arrowprops=dict(facecolor='black', shrink=1))
axs[1].axvline(x=len(mouse8_elbnd[999:])-625, color='k', linestyle=":")
axs[1].set_ylabel('$ELBND$ $[-]$')
axs[1].set_title('Pz', rotation='horizontal', x=1.04, y=0.4)
axs[2].plot(mouse14_elbnd[999:])
axs[2].annotate('seizure', xy=(len(mouse14_elbnd[999:])-625, max(mouse14_elbnd[999:])), xytext=(1200, max(mouse14_elbnd[999:])), arrowprops=dict(facecolor='black', shrink=1))
axs[2].axvline(x=len(mouse14_elbnd[999:])-625, color='k', linestyle=":")
axs[2].set_ylabel('$ELBND$ $[-]$')
axs[2].set_title('Fp1', rotation='horizontal', x=1.04, y=0.4)
plt.xlabel('$k$ $[-]$')
plt.autoscale(enable=True, axis='x', tight=True)



fig, axs = plt.subplots(3, 1, sharex=True)
# fig.subplots_adjust(hspace=0)

axs[0].plot(mouse0_le[999:])
axs[0].annotate('seizure', xy=(len(mouse0_le[999:])-625, max(mouse0_le[999:])), xytext=(1200, max(mouse0_le[999:])), arrowprops=dict(facecolor='black', shrink=1))
axs[0].axvline(x=len(mouse0_le[999:])-625, color='k', linestyle=":")
axs[0].set_ylabel('$LE$ $[-]$')
axs[0].set_title('C3', rotation='horizontal', x=1.04, y=0.4)
axs[1].plot(mouse8_le[999:])
axs[1].annotate('seizure', xy=(len(mouse8_le[999:])-625, max(mouse8_le[999:])), xytext=(1200, max(mouse8_le[999:])), arrowprops=dict(facecolor='black', shrink=1))
axs[1].axvline(x=len(mouse8_le[999:])-625, color='k', linestyle=":")
axs[1].set_ylabel('$LE$ $[-]$')
axs[1].set_title('Pz', rotation='horizontal', x=1.04, y=0.4)
axs[2].plot(mouse14_le[999:])
axs[2].annotate('seizure', xy=(len(mouse14_le[999:])-625, max(mouse14_le[999:])), xytext=(1200, max(mouse14_le[999:])), arrowprops=dict(facecolor='black', shrink=1))
axs[2].axvline(x=len(mouse14_le[999:])-625, color='k', linestyle=":")
axs[2].set_ylabel('$LE$ $[-]$')
axs[2].set_title('Fp1', rotation='horizontal', x=1.04, y=0.4)
plt.xlabel('$k$ $[-]$')

plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

