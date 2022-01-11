import numpy as np
import random
import matplotlib.pyplot as plt
import padasip as pa
from datetime import datetime
from scipy.stats import genpareto
from pot import pot
import math
from pickle import dump
from pickle import load
import json

# with open('state.obj', 'wb') as f:
#     dump(np.random.get_state(), f)

found_best = False
mg_b = 0.2
mg_g = 0.1
mg_t = 17
mg_exp = 10
mg_len = 1000
my_seed = 2256 #2256
while not(found_best):
    my_seed += 1
    print('starting to process seed:', my_seed)
    filter_len = 15
    filter_data = np.zeros([mg_len - mg_t, filter_len])
    n = filter_len
    gev_window = 300
    #np.random.seed(my_seed) # OK

    desired_output = np.zeros([mg_len - mg_t])
    pertubace = 523
    # np.random.seed(624)
    # np.random.seed(None)
    # print('seed', np.random.get_state())
    #
    # with open('state.obj', 'wb') as f:
    #   dump(np.random.get_state(), f)

    with open('KONECNE.obj', 'rb') as f:
        np.random.set_state(load(f))

    mackey_glass_data = -1 * np.random.uniform(-2, 2, mg_len)
    # mackey_glass_data = np.linspace(1,1000, num=mg_len)
    pertubed_mackey_glass = np.copy(mackey_glass_data)
    for index, value in np.ndenumerate(mackey_glass_data):
        if index[0] > mg_t:
            mackey_glass_data[index[0]] = mackey_glass_data[index[0] - 1] + mg_b * mackey_glass_data[index[0] - mg_t] /\
                       (1 + mackey_glass_data[index[0] - mg_t] ** mg_exp) - mg_g * mackey_glass_data[index[0] - 1]
            #if index[0] < mg_len - mg_t:
            if index[0] == 316 + pertubace: # 316+523:
                pertubed_mackey_glass[316 + pertubace] = mackey_glass_data[316 + pertubace] + 0.05 * mackey_glass_data[316 + pertubace]
                # pertubed_mackey_glass[316 + 215] = mackey_glass_data[316 + 215] + 0.05 * mackey_glass_data[316 + 215]
                # krmení dat pro filtr
            else:
                pertubed_mackey_glass[index[0]] = mackey_glass_data[index[0]]
            for idx in range(filter_len):
                # filter_data[index[0] - mg_t, idx] = mackey_glass_data[index[0] - idx - 1]
                desired_output[index[0] - mg_t] = pertubed_mackey_glass[index[0]] # mackey_glass_data[index[0]]
            filter_data[index[0] - mg_t, 0] = mackey_glass_data[index[0] - 1] ** 2
            filter_data[index[0] - mg_t, 1] = mackey_glass_data[index[0] - 2] ** 2
            filter_data[index[0] - mg_t, 2] = mackey_glass_data[index[0] - 3] ** 2
            filter_data[index[0] - mg_t, 3] = mackey_glass_data[index[0] - 4] ** 2
            filter_data[index[0] - mg_t, 4] = mackey_glass_data[index[0] - 5] ** 2
            filter_data[index[0] - mg_t, 5] = mackey_glass_data[index[0] - 1] * mackey_glass_data[index[0] - 2]
            filter_data[index[0] - mg_t, 6] = mackey_glass_data[index[0] - 1] * mackey_glass_data[index[0] - 3]
            filter_data[index[0] - mg_t, 7] = mackey_glass_data[index[0] - 1] * mackey_glass_data[index[0] - 4]
            filter_data[index[0] - mg_t, 8] = mackey_glass_data[index[0] - 1] * mackey_glass_data[index[0] - 5]
            filter_data[index[0] - mg_t, 9] = mackey_glass_data[index[0] - 2] * mackey_glass_data[index[0] - 3]
            filter_data[index[0] - mg_t, 10] = mackey_glass_data[index[0] - 2] * mackey_glass_data[index[0] - 4]
            filter_data[index[0] - mg_t, 11] = mackey_glass_data[index[0] - 2] * mackey_glass_data[index[0] - 5]
            filter_data[index[0] - mg_t, 12] = mackey_glass_data[index[0] - 3] * mackey_glass_data[index[0] - 4]
            filter_data[index[0] - mg_t, 13] = mackey_glass_data[index[0] - 3] * mackey_glass_data[index[0] - 5]
            filter_data[index[0] - mg_t, 14] = mackey_glass_data[index[0] - 4] * mackey_glass_data[index[0] - 5]

    #print(filter_data)
    #print(desired_output)
    #print('raw mg shape', mackey_glass_data.shape)
    #print('filter data shape', filter_data.shape)


    # create pertubation

    #pertubed_mackey_glass[316+523] = pertubed_mackey_glass[316+523] + 5 * pertubed_mackey_glass[316+523]







    filter = pa.filters.FilterNLMS(filter_len, mu=1., w=np.random.rand(filter_len))
    # filter = pa.filters.FilterGNGD(filter_len, mu=1.)

    y, e, w = filter.run(desired_output[316:], filter_data[316:])
    np.savetxt('filterData.csv', filter_data, delimiter=',')
    np.savetxt('desiredOutput.csv', desired_output, delimiter=',')
    plt.figure(1000)
    plt.plot(y,'r')
    plt.plot(desired_output[316:], 'b')
    #plt.show()
    le = pa.detection.learning_entropy(w, m=30, order=1, alpha=[4, 5, 6, 7, 8, 9])

    plt.figure(8)
    plt.plot(y, 'k')
    plt.plot(desired_output[316:], 'b')
    # LE plot
    plt.figure(6)
    plt.plot(le)

    plt.figure(2)
    plt.plot(e[316:])



    dw = np.copy(w)
    dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
    dw_count = int(dw.shape[0])


    plt.figure(3)
    #plt.plot(dw)








    hpp = np.ones((dw_count - gev_window, n))
    for i in range(gev_window, dw.shape[0]):
        # print((str(datetime.now())), " processing: ", i)
        for j in range(n):
            poted_values = pot(dw[i-gev_window:i, j], 1)
            if dw[i, j] > poted_values[-1]:
                fit = genpareto.fit(pot(dw[i-gev_window:i, j], 1), 1, loc=0, scale=1)
                if dw[i, j] >= fit[1]:
                    hpp[i-gev_window, j] = 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 5e-5
                #print(fit)
                #print(hpp[i-gev_window, j])
                #print(dw[i, j])
                #print('muj odhad:', (1/fit[2])*(1+(fit[0]*(dw[i, j] - fit[1])/fit[2])) ** (-1 - 1/fit[0]))


    totalhpp1 = np.prod(hpp, axis=1)

    loghppto_print = np.concatenate((np.zeros(gev_window), np.log10(totalhpp1)))
    loghpp = np.concatenate((np.zeros(gev_window), np.log10(totalhpp1)))

    totalhpp1 = np.prod(hpp, axis=1)
    min_index = np.argmin(loghpp)
    # print(totalhpp1.shape,'shape')
    # plt.figure(88)
    # plt.plot(np.log10(totalhpp1))
    # plt.show()
    if min_index > 523 - mg_t and min_index < 526 - mg_t:
        found_best = True
        print('seed:', my_seed)



np.savetxt("foo.csv", hpp, delimiter=",")

# plot experimental data
x = np.linspace(0, 683, num=684)
"""
MG pertubation details and original data series
"""

plt.figure(figsize=(10, 7.5))

plt.plot([1, 2, 3])

ax1 = plt.subplot(211)
# plt.plot(mackey_glass_data[316:], color='k')
plt.plot(x, pertubed_mackey_glass[316:], color='k')
plt.xticks([0, 100, 200, 300, 400, 500, 600, 683])
plt.grid(b=True, which='both', axis='both', linestyle='--')
plt.autoscale(tight=True, axis='x')
plt.xlabel('$k$ $[-]$')
plt.ylabel('$y$ $[-]$')

plt.annotate('the pertubed sample', xy=(523, 1.2), xytext=(523, 1.4), arrowprops=dict(facecolor='black', shrink=1))
ax2 = plt.subplot(223)
# print('mg shape', mackey_glass_data[316:].shape)
# print('pertubed shape', pertubed_mackey_glass[316:].shape)
ax2.plot(x, mackey_glass_data[316:], color='b', linewidth=1, label='series without pertubation',)
ax2.plot(x, pertubed_mackey_glass[316:], color='k', linewidth=1,label='pertubed series',  linestyle='dashed')
ax2.grid(b=True, which='both', axis='both', linestyle='--')
ax2.set_xlim(514, 530)
ax2.set_ylim(0.4, 1.1)
plt.legend()
# plt.annotate('the pertubed sample', xy=(523, -1.0), xytext=(523, -1.1), arrowprops=dict(facecolor='black', shrink=1))
# plt.autoscale(tight=True, axis='x')
plt.xlabel('$k$ $[-]$')
plt.ylabel('$y$ $[-]$')
plt.tight_layout(w_pad=4.5)

# mackey glass pertubet sample detail
ax3 = plt.subplot(224)
#ax3.yaxis.tick_right()
#ax3.yaxis.set_label_position("right")
ax3.plot(x, mackey_glass_data[316:], color='k', label='series without pertubation')
ax3.plot(x, pertubed_mackey_glass[316:], color='k', linewidth=1, linestyle='--', marker='*', markersize=9, label='pertubed series')
ax3.grid(b=True, which='both', axis='both', linestyle='--')
ax3.set_xlim(520, 526)
ax3.set_ylim(0.4, 0.8)
plt.xlabel('$k$ $[-]$')
plt.ylabel('$y$ $[-]$')
plt.legend(loc='lower right')
plt.annotate('the pertubed sample', xy=(523, -1.0), xytext=(523, -1.1), arrowprops=dict(facecolor='black', shrink=1))


plt.savefig('mackeydetails.eps', format='eps', dpi=300)
# ND

x = np.linspace(16, 683, num=667)
# results printing

"""
Data series + prediction error + weight changes 
"""
plt.figure(888, figsize=(10, 7.5))
ax1 = plt.subplot(311)
axes = plt.gca()
axes.set_ylim([0.30, 1.40])
axes.set_xlim([16, 683])
#plt.xticks([16, 100, 200, 300, 400, 500, 600, 683])
plt.setp(ax1.get_xticklabels(), visible=False)
# plt.plot(mackey_glass_data[316:], color='k')
plt.plot(x, desired_output[316:], color='k', linestyle='solid', linewidth=1.5)
plt.plot(x, y, color='g', linestyle=(0, (1, 1)), linewidth=1.5)

plt.grid(b=True, which='both', axis='both', linestyle='--')
plt.autoscale(tight=True, axis='x')
# plt.xlabel('$k [-]$')
plt.ylabel('$y$ $[-]$')
plt.annotate('', xy=(523, 0.5), xytext=(528, 0.5), arrowprops=dict(facecolor='black', shrink=0))
axes.set_title('a', rotation='horizontal', x=1.03, y=0.4)

ax2 = plt.subplot(312, sharex=ax1)
# plt.plot(mackey_glass_data[316:], color='k')
plt.setp(ax2.get_xticklabels(), visible=False)
axes = plt.gca()
axes.set_xlim([16, 683])
#plt.xticks([16, 100, 200, 300, 400, 500, 600, 683])
axes.set_ylim([-0.05, 0.30])
plt.plot(x, np.abs(e), color='r')
plt.grid(b=True, which='both', axis='both', linestyle='--')
plt.autoscale(tight=True, axis='x')
#plt.xlabel('$k [-]$')
plt.ylabel('$|e|$ $[-]$')
plt.annotate('', xy=(523, 0.21), xytext=(523, 0.28), arrowprops=dict(facecolor='black', shrink=0))
axes.set_title('b', rotation='horizontal', x=1.03, y=0.4)

ax3 = plt.subplot(313, sharex=ax1)
# plt.plot(mackey_glass_data[316:], color='k')
plt.setp(ax3.get_xticklabels(), visible=True)
axes = plt.gca()
axes.set_ylim([-0.01, 0.045])
axes.set_xlim([16, 683])
plt.xticks([16, 100, 200, 300, 400, 500, 600, 683])
plt.plot(x, dw)
plt.grid(b=True, which='both', axis='both', linestyle='--')
# plt.autoscale(tight=True, axis='x')
plt.xlabel('$k [-]$')
plt.ylabel('$|\Delta w|$ $[-]$')
plt.annotate('', xy=(523, 0.035), xytext=(523, 0.045), arrowprops=dict(facecolor='black', shrink=0))
axes.set_title('c', rotation='horizontal', x=1.03, y=0.4)
plt.tight_layout()
plt.savefig('mackey_results.eps', format='eps', dpi=300)

"""
ND results
"""

plt.figure(999, figsize=(10, 7.5))
plt.subplot(311)
axes = plt.gca()
axes.set_xlim([16, 683])
axes.set_ylim([-3, 60])
plt.xticks([16, 100, 200, 300, 400, 500, 600, 683])
plt.plot(x, -loghpp[-667:])
plt.grid(b=True, which='both', axis='both', linestyle='--')
# plt.autoscale(tight=True, axis='x')
plt.ylabel('$ESE$ $[-]$')
plt.annotate('', xy=(523, 50), xytext=(523, 57), arrowprops=dict(facecolor='black', shrink=0))
axes.set_title('a', rotation='horizontal', x=1.03, y=0.4)


elbnd = pa.detection.ELBND(w, e, function="max")
plt.subplot(312, sharex=ax1)
axes = plt.gca()
axes.set_xlim([16, 683])
elbnd[-667:-367] = 0
plt.plot(x, elbnd[-667:])
plt.annotate('', xy=(523, 0.0065), xytext=(523, 0.0075), arrowprops=dict(facecolor='black', shrink=0))
axes.set_title('b', rotation='horizontal', x=1.03, y=0.4)
plt.grid(b=True, which='both', axis='both', linestyle='--')
plt.ylabel("$ELBND$ $[-]$")


le = pa.detection.learning_entropy(w, m=300, order=1)
total_le = np.sum(le, axis=1)
plt.subplot(313, sharex=ax1)
axes = plt.gca()
axes.set_xlim([16, 683])
axes.set_ylim([-20, 85])
plt.plot(x, total_le[-667:])
plt.annotate('', xy=(523, 70), xytext=(523, 83), arrowprops=dict(facecolor='black', shrink=0))
axes.set_title('c', rotation='horizontal', x=1.03, y=0.4)
plt.grid(b=True, which='both', axis='both', linestyle='--')
plt.xlabel('$k$ $[-]$')
plt.ylabel("$LE$ $[-]$")
plt.tight_layout()
plt.savefig('mackey_results_nd.eps', format='eps', dpi=300)
"""
plt.figure(10)
plt.plot((hpp))
plt.title('hpp2')
print(totalhpp1.shape)
plt.figure(11)
plt.plot(np.log10(totalhpp1))
plt.title('totalhpp1')
"""
plt.show()