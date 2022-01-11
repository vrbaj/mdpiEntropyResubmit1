import padasip as pa
import numpy as np
from pot import pot
from datetime import datetime
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import pickle
from colorama import Fore, Back, Style


experiments_number = 1000
experiment_len = 1600
inputs_number = 2
filter_len = 3
parameter_change_idx = 1400
gev_window = 1200
noise_sigma = 0.08

gpd_result = np.zeros([experiment_len, ])
elbnd_result = np.zeros([experiment_len, ])
le_result = np.zeros([experiment_len, ])
e_result = np.zeros([experiment_len, ])
snr = np.zeros([experiments_number, ])


for seed_counter in range(0, experiments_number):
    np.random.seed(seed_counter)
    x = np.random.uniform(low=-1, high=1, size=(experiment_len, inputs_number))
    # x = np.random.normal(0,1, size=(experiment_len, inputs_number))
    desired_output = np.zeros([experiment_len, ])
    filter_data = np.zeros([experiment_len, 3])
    random_weights = np.random.uniform(low=-1, high=1, size=3)
    # random_weights = np.random.normal(0, 1, 3)
    noiseless_signal = np.zeros([experiment_len, ])
    for idx in range(experiment_len):
        if idx == 0 or idx == parameter_change_idx:
            # random_weights = np.random.normal(0, 1, 3)
            random_weights = np.random.uniform(low=-1, high=1, size=3)
        filter_data[idx, 0] = x[idx, 0]
        filter_data[idx, 1] = x[idx, 1]
        filter_data[idx, 2] = x[idx, 0] * x[idx, 1]
        if idx < parameter_change_idx:
            desired_output[idx] = random_weights[0] * x[idx, 0] + random_weights[1] * x[idx, 1] + random_weights[2] * x[idx, 0] * x[idx, 1] + np.random.normal(0, noise_sigma, 1)
            noiseless_signal[idx] = random_weights[0] * x[idx, 0] + random_weights[1] * x[idx, 1] + random_weights[2] * x[idx, 0] * x[idx, 1]
        else: # 0.4 1.6
            desired_output[idx] = random_weights[0] * x[idx, 0] + random_weights[1] * x[idx, 1] + random_weights[2] * x[idx, 0] * x[idx, 1] \
                                  + np.random.normal(0, noise_sigma, 1)
            noiseless_signal[idx] = random_weights[0] * x[idx, 0] + random_weights[1] * x[idx, 1] + random_weights[2] * x[idx, 0] * x[idx, 1]

    honu_filter = pa.filters.FilterGNGD(filter_len, mu=1.)
    y, e, w = honu_filter.run(desired_output, filter_data)
    elbnd = pa.detection.ELBND(w, e, function="sum")

    dw = np.copy(w)
    dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
    dw_count = int(dw.shape[0])

    hpp = np.ones((dw_count - gev_window, filter_len))
    for i in range(gev_window, dw.shape[0]):
        if i % 100 == 0:
            pass  # print((str(datetime.now())), " processing: ", i)
        for j in range(filter_len):
            poted_values = pot(dw[i - gev_window:i, j], 1)

            if dw[i, j] > poted_values[-1]:
                fit = genpareto.fit(poted_values, floc=[poted_values[-1]])
                if dw[i, j] >= fit[1]:
                    hpp[i - gev_window, j] = 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 1e-50
    totalhpp1 = np.prod(hpp, axis=1)
    min_index = np.argmin(totalhpp1)
    snr[seed_counter] = 10 * np.log10((np.std(desired_output[gev_window:]) ** 2) / (noise_sigma ** 2))
    print(Fore.RED + "experiment number: " + str(seed_counter))
    print(Fore.GREEN + "SNR: " + (str(snr[seed_counter])))
    print(Fore.BLACK + "min_index GPD: " + str(min_index))
    if min_index > 199 and min_index < 211:
        gpd_result[seed_counter] = 1

    max_index_elbnd = np.argmax(elbnd[-401:])
    print("max_index elbnd: ", max_index_elbnd)
    if max_index_elbnd > 199 and max_index_elbnd < 211:
        elbnd_result[seed_counter] = 1

    le = pa.detection.learning_entropy(w, m=1200, order=1)
    total_le = np.sum(le, axis=1)
    max_index_le = np.argmax(total_le[-400:])
    print("max_index LE: ", max_index_le)
    if max_index_le > 199 and max_index_le < 211:
        le_result[seed_counter] = 1

    max_index_e = np.argmax(abs(e[-400:]))
    print("max_index E: ", max_index_e)
    if max_index_e > 199 and max_index_e < 211:
        e_result[seed_counter] = 1

    print("GPD detections: ", sum(gpd_result) / (seed_counter + 1)) #experiments_number)
    print("ELBND detections: ", sum(elbnd_result) / (seed_counter + 1)) #experiments_number)
    print("LE detections: ", sum(le_result) / (seed_counter + 1))#experiments_number)
    print("E detections: ", sum(e_result) / (seed_counter + 1)) #experiments_number)
    print("AVG SNR: ", sum(snr) / (seed_counter + 1))

print("GPD detections: ", sum(gpd_result) /experiments_number)
print("ELBND detections: ", sum(elbnd_result) / experiments_number)
print("LE detections: ", sum(le_result) / experiments_number)
print("E detections: ", sum(e_result) / experiments_number)
print("AVG SNR: ", np.mean(snr))













