import numpy as np

Z_data = np.loadtxt('bonnZtotalhpp.txt')
S_data = np.loadtxt('bonnStotalhpp.txt')
O_data = np.loadtxt('bonnOtotalhpp.txt')

print('Z sum: ', np.sum(Z_data))
print('S sum: ', np.sum(S_data))
print('O sum: ', np.sum(O_data))

print('Z std: ', np.std(Z_data))
print('S std: ', np.std(S_data))
print('O std: ', np.std(O_data))

print('Z max: ', np.min(Z_data))
print('S max: ', np.min(S_data))
print('O max: ', np.min(O_data))
