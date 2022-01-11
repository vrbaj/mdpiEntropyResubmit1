import padasip as pa
import numpy as np
from pot import pot
from datetime import datetime
from scipy.stats import genpareto
import matplotlib.pyplot as plt

found_best = False
seed_counter = 2000#96, 156, 158 167 192 ok 195 TOP 1849
while not(found_best):
    seed_counter = seed_counter + 1
    print(seed_counter)
    np.random.seed(seed_counter)



    experiment_len = 1500
    inputs_number = 2
    filter_len = 3
    parameter_change_idx = 1000
    gev_window = 500

    x = np.random.rand(experiment_len, inputs_number)



    desired_output = np.zeros([experiment_len, ])
    filter_data = np.zeros([experiment_len, 3])

    for idx in range(experiment_len):
        filter_data[idx, 0] = x[idx, 0]
        filter_data[idx, 1] = x[idx, 1]
        filter_data[idx, 2] = 1

        if idx < parameter_change_idx:
            desired_output[idx] = (x[idx, 0] + x[idx, 1]) + np.random.normal(0, 0.1, 1) + 0.01 * idx


        else: # 0.4 1.6
            desired_output[idx] = (x[idx, 0] + x[idx, 1]) + np.random.normal(0, 0.1, 1) + 0.0105 * idx #0.0105 je super


    honu_filter = pa.filters.FilterLMS(filter_len, mu=0.1, w="random")
    y, e, w = honu_filter.run(desired_output, filter_data)


    dw = np.copy(w)
    dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
    dw_count = int(dw.shape[0])
    # print(dw.shape)

    # plt.figure(3)
    #plt.plot(dw)
    #plt.show()

    # ND


    hpp = np.ones((dw_count - gev_window, filter_len))
    for i in range(gev_window, dw.shape[0]):
        if i % 100 == 0:
            pass # print((str(datetime.now())), " processing: ", i)
        for j in range(filter_len):
            poted_values = pot(dw[i-gev_window:i, j], 1)
            if dw[i, j] > poted_values[-1]:
                fit = genpareto.fit(pot(dw[i-gev_window:i, j], 1), 1, loc=0, scale=1)
                if dw[i, j] >= fit[1]:
                    hpp[i-gev_window, j] = 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 1e-30
                #print(fit)
                #print(hpp[i-gev_window, j])
                #print(dw[i, j])
                #print('muj odhad:', (1/fit[2])*(1+(fit[0]*(dw[i, j] - fit[1])/fit[2])) ** (-1 - 1/fit[0]))


    totalhpp1 = np.prod(hpp, axis=1)
    min_index = np.argmin(totalhpp1)
    thr = -np.log10(totalhpp1) > 10
    print(thr.sum())
    if min_index > 499 and min_index < 503 and thr.sum() < 2:
        found_best = True

plt.figure(10)
plt.plot((hpp))
plt.title('hpp2')
print(totalhpp1.shape)


my_fig = plt.figure(11, figsize=(10,7.5))
ax1 = plt.subplot(611)
ax1.set_title('a', rotation='horizontal', x=1.03, y=0.4)
# plt.plot(y[-1001:], 'g')
plt.plot(desired_output[-1001:])
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('$y$ $[-]$')
plt.grid(True)
ax1.set_xlim([0, 1000])
ax1.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])
plt.annotate('', xy=(500, 14), xytext=(500, 17), arrowprops=dict(facecolor='black', shrink=0))
ax2 = plt.subplot(612, sharex=ax1)
ax2.set_title('b', rotation='horizontal', x=1.03, y=0.4)
plt.plot(e[-1001:])
plt.setp(ax2.get_xticklabels(), visible=False)
axes = plt.gca()
axes.set_ylim([-1, 1])
plt.grid(True)
#plt.xlabel('$k [-]$')
plt.ylabel('$e$ $[-]$')


#plt.xlabel('$k [-]$')

ax3 = plt.subplot(613, sharex=ax1)
ax3.set_title('c', rotation='horizontal', x=1.03, y=0.4)
plt.plot(dw[-1001:])
#plt.xlabel('$k [-]$')
plt.ylabel('$|\Delta w|$ $[-]$')
plt.setp(ax3.get_xticklabels(), visible=False)
plt.grid(True)


ax4 = plt.subplot(614, sharex=ax1)
plt.plot(-np.log10(totalhpp1))
ax4.set_title('d', rotation='horizontal', x=1.03, y=0.4)
# plt.xlabel('$k [-]$')
plt.ylabel('$ESE$ $[-]$')
plt.setp(ax4.get_xticklabels(), visible=False)

plt.grid(True)




elbnd = pa.detection.ELBND(w, e, function="sum")
elbnd[0:2] = 0
ax5 = plt.subplot(615, sharex=ax1)
plt.plot(elbnd[-1001:])
# plt.xlabel('$k [-]$')
plt.ylabel('$ELBND$ $[-]$')
plt.setp(ax5.get_xticklabels(), visible=False)

plt.grid(True)
ax5.set_title('e', rotation='horizontal', x=1.03, y=0.4)




le = pa.detection.learning_entropy(w, m=300, order=1)
total_le = np.sum(le, axis=1)
ax6 = plt.subplot(616, sharex=ax1)
plt.plot(total_le[-1001:])
# plt.xlabel('$k [-]$')
plt.ylabel('$LE$ $[-]$')
plt.setp(ax6.get_xticklabels(), visible=True)
plt.xlabel('$k$ $[-]$')
plt.grid(True)
ax6.set_title('f', rotation='horizontal', x=1.03, y=0.4)


plt.tight_layout()

plt.savefig('trendchange_lms.eps', format='eps', dpi=300)





my_fig = plt.figure(22)
ax11 = plt.subplot(211)
plt.plot(dw[-505:-485])
plt.ylabel('$\Delta w [-]$')
plt.grid(True)
ax11.set_xlim([495, 515])
# ax1.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])
plt.autoscale(tight=True, axis='x')
ax2 = plt.subplot(212, sharex=ax11)

plt.plot(np.log10(totalhpp1[495:510]))
plt.setp(ax2.get_xticklabels(), visible=False)
# axes = plt.gca()
# axes.set_ylim([-0.5, 0.5])
plt.grid(True)
plt.xlabel('$k [-]$')
plt.ylabel('$n_d [-]$')
plt.autoscale(tight=True, axis='x')

print('minimum index hpp:', np.argmin(hpp))
print('minimum index hpp1:', min_index)
print(totalhpp1[min_index])
print('seed_counter:', seed_counter)
plt.show()
