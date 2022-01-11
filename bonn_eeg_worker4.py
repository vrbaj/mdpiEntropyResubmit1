import numpy as np
import matplotlib.pyplot as plt
import padasip as pa
from datetime import datetime
from scipy.stats import genpareto
from pot import pot


data_set_letter = 'S'
file_to_process = 'bonn_' + data_set_letter + '.txt'

raw_data = np.loadtxt(file_to_process)
raw_data = raw_data[::4]
raw_data_size = raw_data.shape[0]
plt.plot(raw_data[::10], linewidth=0.2)
plt.show()


filter_len = 10
desired_output = np.zeros(raw_data_size-filter_len)
filter_data = np.zeros([raw_data_size-filter_len, filter_len])

filter = pa.filters.FilterNLMS(filter_len, mu=1., w=np.random.rand(filter_len))
# filter = pa.filters.FilterGNGD(filter_len, mu=1.)
counter = 0
counter2 = 0
for idx, sample in np.ndenumerate(raw_data):
    if counter >= 6 * filter_len:
        desired_output[counter2] = sample
        for i in range(filter_len):

            filter_data[counter2, i] = raw_data[idx[0] - 1 - i * 2]
        counter2 = counter2 + 1
    counter = counter + 1

y, e, w = filter.run(desired_output, filter_data)
dw = np.copy(w)
dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
dw_count = int(dw.shape[0])


gev_window = 3000

hpp = np.ones((dw_count - gev_window, filter_len))
for i in range(gev_window, dw.shape[0]):
    if i % 100 == 0:
        print((str(datetime.now())), " processing: ", i)
    for j in range(filter_len):
        poted_values = pot(dw[i-gev_window:i, j], 1)
        if dw[i, j] > poted_values[-1]:
            fit = genpareto.fit(pot(dw[i-gev_window:i, j], 1), 1, loc=0, scale=1)
            if dw[i, j] >= fit[1]:
                hpp[i-gev_window, j] = 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 5e-100
            #print(fit)
            #print(hpp[i-gev_window, j])
            #print(dw[i, j])
            #print('muj odhad:', (1/fit[2])*(1+(fit[0]*(dw[i, j] - fit[1])/fit[2])) ** (-1 - 1/fit[0]))


totalhpp1 = np.prod(hpp, axis=1)

plt.figure(10)
plt.plot((hpp))
plt.title('hpp2' + data_set_letter)
print(totalhpp1.shape)
plt.figure(11)
plt.plot(np.log10(totalhpp1))
plt.title('totalhpp1' + data_set_letter)


min_index = np.argmin(totalhpp1)
print('minimum index hpp:', np.argmin(hpp))
print('minimum index hpp1:', min_index)
print(totalhpp1[min_index])

plt.show()

np.savetxt('bonn' + data_set_letter + 'totalhpp.txt', totalhpp1)
