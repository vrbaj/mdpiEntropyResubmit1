import sys
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np

import scipy.io

# import
############################


### epilepsy clear
filename = "8507143722.mat"
start = 100
end = 152
issue = "epilpsy1"

## epilepsy clear
# filename = "0359233259.mat"
# start = 575
# end = 595
# issue = "epilpsy2"

### muscle activity
# filename = "0359233259.mat"
# start = 465
# end = 485
# issue = "musle_activity"


mat = scipy.io.loadmat(filename)["dz"][:19]
print(mat.shape)
labels = (
'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2')

# plt.figure(figsize=(8, 11))

shift1 = 4

N1 = len(mat[1][start * 128 + (end - start) * 64:end * 128:1])

#t1 = np.array(range(N1)) / 128.  # + start + (end-start)/2

YRM = np.zeros((3, N1))
help_idx=0
for i, line in enumerate(mat[::-1]):
    if i in (0,8,14):
        yr_plot = line[start * 128 + (end - start) * 64:end * 128:1]

        yr_plot = (yr_plot - np.mean(yr_plot)) / 3. / np.std(yr_plot)

        YRM[help_idx, :] = yr_plot + shift1 * (help_idx + 1)
        help_idx += 1



font = {'size': 10}
plt.rc('font', **font)

y_labels = [np.mean(YRM[-i, :]) for i in (0, 1, 2)]

plt.yticks(y_labels, ('Fp1','C3','Pz'))

print
#len(t1), len(YRM)
print(YRM.shape)
plt.plot(YRM.T[1000:,:],'C0')
plt.grid(True)
plt.xlabel("k [-]")
plt.annotate('seizure', xy=(len(YRM.T[1000:,2]) - 625, 2.5 + min(YRM.T[:,2])), xytext=(1300, 2.6 + min(YRM.T[:,2])), arrowprops=dict(facecolor='black', shrink=1))
plt.annotate('seizure', xy=(len(YRM.T[1000:,1]) - 625,  0.2 + min(YRM.T[:,1])), xytext=(1300, min(YRM.T[:,1]+0.1)), arrowprops=dict(facecolor='black', shrink=1))
plt.annotate('seizure', xy=(len(YRM.T[1000:,0]) - 625, 0.2 + min(YRM.T[:,0])), xytext=(1300, min(YRM.T[:,0]+0.1)), arrowprops=dict(facecolor='black', shrink=1))
#plt.plot( YRM.T, "b")
#plt.xlim(t1[0], t1[-1])
# plt.figure(333)
# plt.plot(YRM.T[:, 13])
# np.savetxt('eeg.csv', YRM.T)
# savename = ""
# # plt.savefig('fig-{}{}.png'.format(issue, savename), dpi=300, bbox_inches='tight')
# print(t1.shape)
plt.tight_layout()
print("seizure start:", (len(YRM.T[1000:,0]) - 625))

plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig('mouse_eeg.eps', format='eps', dpi=300)
plt.show()











