import scipy.io
import os.path
import matplotlib.pyplot as plt
import numpy as np


mat_path = os.path.join('Dog_1', 'Dog_1_preictal_segment_0007.mat')
mat = scipy.io.loadmat(mat_path)
print(list(mat.keys()))
print(mat)
some_data = (mat['preictal_segment_7'][0][0][0][1])
some_data = (some_data[:] - np.mean(some_data))/np.std(some_data)
print(some_data.shape)

# plt.figure(1)
# plt.plot(some_data)
#
#
# plt.show()
