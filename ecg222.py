import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import padasip as pa
from datetime import datetime
from scipy.stats import genpareto
from pot import pot

np.random.seed(10)

ecg_data = np.genfromtxt('222.csv', delimiter=',', names=True)
print(ecg_data['V1'])
print(ecg_data.shape)


ecg_data_raw = ecg_data['V1'][370001:380001]
raw_data_size = ecg_data_raw.shape[0]
# ecg_data_raw = (ecg_data_raw[:] - np.mean(ecg_data_raw)) / np.std(ecg_data_raw) / 3
counter = 0
step_size = 1000
resampled_len = int(raw_data_size / step_size)
print('resampled len', resampled_len)
resampled_ecg = np.zeros(resampled_len,)
for sample in ecg_data_raw[:-step_size:step_size]:
    # print(ecg_data_raw[counter*step_size:counter*step_size+step_size])
    resampled_ecg[counter] = np.mean((ecg_data_raw[counter*step_size:counter*step_size+step_size]))
    counter = counter + 1

plt.figure(1)
plt.plot(ecg_data_raw[:])
plt.figure(2)
plt.plot(resampled_ecg[:], 'r')
plt.show()


filter_len = 15
desired_output = np.zeros(raw_data_size-filter_len)
filter_data = np.zeros([raw_data_size-filter_len, filter_len])

filter = pa.filters.FilterNLMS(filter_len, mu=1., w=np.random.rand(filter_len))
# filter = pa.filters.FilterGNGD(filter_len, mu=1.)
counter = 0
counter2 = 0
for idx, sample in np.ndenumerate(ecg_data_raw):
    if counter >= 5 * filter_len:
        desired_output[counter2] = sample
        for i in range(filter_len):

            filter_data[counter2, i] = ecg_data_raw[idx[0] - 1 - i * 5]
        counter2 = counter2 + 1
    counter = counter + 1




y, e, w = filter.run(desired_output, filter_data)
np.savetxt('filterDataECG.csv', filter_data, delimiter=',')
np.savetxt('desiredOutputECG.csv', desired_output, delimiter=',')



dw = np.copy(w)
dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
dw_count = int(dw.shape[0])
print(dw.shape)

plt.figure(3)
plt.plot(dw)
plt.show()

# ND
gev_window = 500

hpp = np.ones((dw_count - gev_window, filter_len))
for i in range(gev_window, dw.shape[0]):
    if i % 100 == 0:
        print((str(datetime.now())), " processing: ", i)
    for j in range(filter_len):
        poted_values = pot(dw[i-gev_window:i, j], 3)
        # poted_values = pot(dw[5:i, j], 1)
        if dw[i, j] > poted_values[-1]:
            # fit = genpareto.fit(pot(dw[i-gev_window:i, j], 2), 1, loc=0, scale=1)
            # fit = genpareto.fit(pot(dw[2:i, j], 3), 1, loc=0, scale=1)
            fit = genpareto.fit(poted_values, 1, loc=0.1, scale=1)
            if dw[i, j] >= fit[1]:
                hpp[i-gev_window, j] = genpareto.sf(dw[i, j], fit[0], fit[1], fit[2]) + 1e-10 # 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 5e-20
            #print(fit)
            #print(hpp[i-gev_window, j])
            #print(dw[i, j])
            #print('muj odhad:', (1/fit[2])*(1+(fit[0]*(dw[i, j] - fit[1])/fit[2])) ** (-1 - 1/fit[0]))


print('posledni index i', i)

totalhpp1 = np.prod(hpp, axis=1)

plt.figure(10)
plt.plot((hpp))
plt.title('hpp2')
print(totalhpp1.shape)
plt.figure(11)
plt.plot(np.log10(totalhpp1))
plt.title('totalhpp1')


min_index = np.argmin(totalhpp1)
print('minimum index hpp:', np.argmin(hpp))
print('minimum index hpp1:', min_index)
print(totalhpp1[min_index])

plt.show()

np.savetxt("fooECG.csv", hpp, delimiter=",")
