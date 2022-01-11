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

plt.figure(figsize=(8, 11))

shift1 = 4

N1 = len(mat[1][start * 128 + (end - start) * 64:end * 128:1])

t1 = np.array(range(N1)) / 128.  # + start + (end-start)/2

YRM = np.zeros((19, N1))

for i, line in enumerate(mat[::-1]):
    yr_plot = line[start * 128 + (end - start) * 64:end * 128:1]

    yr_plot = (yr_plot - np.mean(yr_plot)) / 3. / np.std(yr_plot)

    YRM[i, :] = yr_plot  + shift1 * (i + 1)



font = {'size': 10}
plt.rc('font', **font)

y_labels = [np.mean(YRM[-i, :]) for i in range(1, 20)]

plt.yticks(y_labels, labels)

print
len(t1), len(YRM)
print(YRM.shape)
#plt.plot(t1, YRM.T, "b")
plt.plot(YRM.T, "b")
#plt.xlim(t1[0], t1[-1])
plt.figure(333)
plt.plot(YRM.T[:, 13])
np.savetxt('eeg.csv', YRM.T)
savename = ""
# plt.savefig('fig-{}{}.png'.format(issue, savename), dpi=300, bbox_inches='tight')
print(t1.shape)
plt.show()










