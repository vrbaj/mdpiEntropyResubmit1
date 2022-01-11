import numpy as np
import matplotlib.pylab as plt
import padasip as pa
from scipy.stats import genextreme

# data creation
np.random.seed(10)
n = 2
N = 20000
# originalni verze DAT:
x = np.random.normal(0, 1, (N, n))
d = np.sum(x, axis=1) + np.random.normal(0, 0.1, N)


# perturbation insertion
d[1618] += 0*0.5
d[5728] +=0*0.3
d[10000] += 0.5 #0.9 #0.5 TOP

# creation of learning model (adaptive filter)
f = pa.filters.FilterNLMS(n, mu=1., w=np.ones(n))
y, e, w = f.run(d, x)
# estimation of LE with weights from learning model
le = pa.detection.learning_entropy(w, m=30, order=1, alpha=[3., 2.,4.,7., 9.])

plt.figure(2)
plt.plot(w[:, 0])
plt.title('w0')





##################################
# EVT APLIKACE
# pouzij prvnich 850 dat na zjisteni parametru

w_pokus = w[1:10001, 1]
print('SELEKCE VAHY:', w_pokus.shape)
fit = genextreme.fit(w_pokus[1:8999])
print('FIT:', fit)

hpp = genextreme.pdf(w_pokus, fit[0], loc=fit[1], scale=fit[2])*fit[2]





print('minimum:', min(hpp[0:10001]))
print('minimum index:', np.argmin(hpp[0:10001]))
##################################

#DW DW DW DW DW DW DW DW DW
dw = np.copy(w)
dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
dw = dw[:,0]
fit2 = genextreme.fit(dw[10:9999])
print('FIT2:', fit2)
hpp2 = genextreme.pdf(dw[1:11001], fit2[0], loc=fit2[1], scale=fit2[2])*fit2[2]
print('odhad hpp2:')
print('minimum2:', min(hpp2))
print('minimum index2:', np.argmin(hpp2))

plt.figure(6)
plt.plot(dw)
plt.title('dw')

plt.figure(7)
plt.plot(d)
plt.title('output')


plt.figure(5)
plt.plot((hpp2))
plt.title('hpp2')




dw2 = np.copy(w)
dw2[1:] = np.abs(np.diff(dw2, n=1, axis=0))
dw2 = dw2[:,1]
fit3 = genextreme.fit(dw2[10:9999])
print('FIT3:', fit3)
hpp3 = genextreme.pdf(dw2[1:11001], fit3[0], loc=fit3[1], scale=fit3[2])*fit3[2]

totalhpp=np.multiply(hpp2,hpp3)
print('minimum total:', min(totalhpp))
print('minimum total:', np.argmin(totalhpp))

# jointed PDF
plt.figure(12)
plt.plot(np.log10(totalhpp), c='k', linewidth=0.8)
plt.xlabel('$k [-]$')
plt.ylabel('$log(f(|\Delta h(k)|))$')
plt.autoscale(tight=True, axis='x')
plt.annotate('disturbed sample detection', xy=(9950, -5.1), xytext=(5000, -5.1),
             arrowprops=dict(facecolor='black', shrink=1))
# filter output
plt.figure(3)
plt.plot(y[1:11001], c='k', linewidth=0.8)
plt.xlabel('$k [-]$')
plt.ylabel('$y [-]$')
plt.autoscale(tight=True, axis='x')
# LE
plt.figure(14)
plt.plot(le[1:11001], c='k', linewidth=0.8)
plt.xlabel('$k [-]$')
plt.ylabel('$LE [-]$')
plt.autoscale(tight=True, axis='both')
plt.annotate('disturbed sample detection', xy=(9950, 0.68), xytext=(5000, 0.65),
             arrowprops=dict(facecolor='black', shrink=1))
# histogram zmen vah
plt.figure(4)
plt.hist(dw[1:11001], bins='auto',color='k')
plt.xlabel('|$\Delta h_1$|')
plt.ylabel('frequency')
plt.autoscale(tight=True, axis='x')
#error
plt.figure(10)
plt.plot(e[1:11001],c='k', linewidth=0.8)
plt.xlabel('$k [-]$')
plt.ylabel('$e [-]$')
plt.autoscale(tight=True, axis='x')
plt.show()