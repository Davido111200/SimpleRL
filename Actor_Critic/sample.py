import numpy as np

a = [[1, 2, 3], [4,5,6]]
print(np.mean(np.array(a), axis=0))
print(type(list(np.mean(np.array(a), axis=0))))