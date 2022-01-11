import numpy as np
import random
import matplotlib.pyplot as plt
import padasip as pa


mg_b = 0.2
mg_g = 0.1
mg_t = 17
mg_exp = 10
mg_len = 10000
training_len = 5000

filter_len = 2
filter_data = np.zeros([mg_len, filter_len])

noise_mean = 0
noise_std = 0.1
noise_length = mg_len - training_len + mg_t


# np.random.seed(30)
mackey_glass_data = np.random.rand(mg_len + mg_t)
for index, value in np.ndenumerate(mackey_glass_data):
    if index[0] > mg_t:
        mackey_glass_data[index[0]] = mackey_glass_data[index[0] - 1] + mg_b * mackey_glass_data[index[0] - mg_t] /\
                    (1 + mackey_glass_data[index[0] - mg_t] ** mg_exp) - mg_g * mackey_glass_data[index[0] - 1]
        if index[0] < mg_len - mg_t:
            filter_data[index[0], 0] = mackey_glass_data[index[0] - 1]
            filter_data[index[0], 1] = mackey_glass_data[index[0] - mg_t]

add_noise = np.random.normal(noise_mean, noise_std, size=noise_length)
mackey_glass_data[training_len:] = mackey_glass_data[training_len:] + add_noise

print(filter_data.shape)
print(mackey_glass_data[mg_t:].shape)

filter = pa.filters.FilterNLMS(filter_len, mu=1., w=np.random.rand(filter_len))
# filter = pa.filters.FilterGNGD(filter_len, mu=1.)
y, e, w = filter.run(mackey_glass_data[mg_t:], filter_data)


plt.figure(1)
plt.plot(mackey_glass_data)

plt.figure(2)
plt.plot(w)



dw = np.copy(w)
dw[1:] = np.abs(np.diff(dw, n=1, axis=0))

plt.figure(3)
plt.plot(dw)
plt.show()


print(mackey_glass_data.shape)
print(mackey_glass_data)

