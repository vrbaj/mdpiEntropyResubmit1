import scipy.io
import os.path
import matplotlib.pyplot as plt
import numpy as np
import padasip as pa
from datetime import datetime
from scipy.stats import genpareto
from pot import pot
from pickle import dump
from pickle import load


#with open('state_mouse.obj', 'wb') as f:
#    dump(np.random.get_state(), f)
with open('state_mouse15.obj', 'rb') as f:
    np.random.set_state(load(f))

filter_len = 10
eeg_data = np.loadtxt('eeg.csv')
# data_raw = eeg_data[:,0] nejlepsi vysledky na svete
counter2 = 0
data_raw = eeg_data[:, 15]
desired_output = np.zeros(eeg_data.shape[0]-filter_len)
filter_data = np.zeros([eeg_data.shape[0]-filter_len, filter_len])


for idx, sample in np.ndenumerate(data_raw):
    if idx[0] >= filter_len:

        desired_output[counter2] = sample
        for i in range(filter_len):

            filter_data[counter2, i] = data_raw[idx[0] - 1 - i]
        counter2 = counter2 + 1

filter = pa.filters.FilterNLMS(filter_len, mu=1., w=np.random.rand(filter_len))
# filter = pa.filters.FilterGNGD(filter_len, mu=1.)
y, e, w = filter.run(desired_output, filter_data)
np.savetxt('filterDataECG.csv', filter_data, delimiter=',')
np.savetxt('desiredOutputECG.csv', desired_output, delimiter=',')



dw = np.copy(w)
dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
# dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
dw_count = int(dw.shape[0])
print(dw.shape)


# ND
gev_window = 1000
# 1000 nejlepsi vysledky na svete

hpp = np.ones((dw_count - gev_window, filter_len))
for i in range(gev_window, dw.shape[0]):
    if i % 100 == 0:
        print((str(datetime.now())), " processing: ", i)
    for j in range(filter_len):
        poted_values = pot(dw[i-gev_window:i, j], 3)
        if dw[i, j] > poted_values[-1]:
            fit = genpareto.fit(poted_values, 1, loc=0, scale=1)
            # print(fit[:])
            if dw[i, j] > fit[1]:
                hpp[i-gev_window, j] = genpareto.sf(dw[i, j], fit[0], fit[1], fit[2]) + 1e-20 #1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 5e-50
            #print(fit)
            #print(hpp[i-gev_window, j])
            #print(dw[i, j])
            #print('muj odhad:', (1/fit[2])*(1+(fit[0]*(dw[i, j] - fit[1])/fit[2])) ** (-1 - 1/fit[0]))


totalhpp1 = np.prod(hpp, axis=1)

plt.figure(10)
plt.plot((hpp))
plt.title('hpp2')
print(totalhpp1.shape)
plt.figure(11)
plt.plot(np.log10(totalhpp1), 'k')
plt.title('totalhpp1')
plt.annotate('the pertubed sample', xy=(len(totalhpp1) - 625, -5.0), xytext=(550, -2), arrowprops=dict(facecolor='red', shrink=1))
plt.savefig('fig-totalhpp1.png', dpi=300, bbox_inches='tight')
np.savetxt('mouse15.csv', -np.log10(totalhpp1))


min_index = np.argmin(totalhpp1)
print('minimum index hpp:', np.argmin(hpp))
print('minimum index hpp1:', min_index)
print(totalhpp1[min_index])


plt.figure(777)
# t1 = np.array(range(data_raw.shape[0]))#  / 128.
plt.plot(data_raw[499:])

plt.figure(888)
elbnd = pa.detection.ELBND(w, e, function="max")
axes = plt.gca()
plt.plot(elbnd[2:])
axes.set_title('b', rotation='horizontal', x=1.03, y=0.4)
plt.grid(b=True, which='both', axis='both', linestyle='--')
plt.ylabel("$ELBND$ $[-]$")

plt.figure(999)
le = pa.detection.learning_entropy(w, m=1000, order=1)
total_le = np.sum(le, axis=1)
plt.plot(total_le)
plt.grid(b=True, which='both', axis='both', linestyle='--')
plt.show()
np.savetxt('mouse15_elbnd.csv', elbnd)
np.savetxt('mouse15_le.csv', total_le)