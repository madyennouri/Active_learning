import numpy as np
def compute_result(x, y):
    z = np.sin(x**2) * np.cos(y**2) + np.exp(-((x - 1)**2 + (y + 2)**2)) * np.sin(3 * x) * np.cos(3 * y) + 0.5 * x * y
    return z

# sum = 0
# num_epoch = 10
# for i in range(num_epoch):
#     sum += np.floor(i+1/10)
#     print(np.floor(i+1/10))
#
# print(sum)