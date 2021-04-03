# This file is to test the usage of some functions in Python
import numpy as np

A = np.array([[0],
              [1],
              [2]])


B = np.array([1, 2, 3])

idx = [2, 1, 1]
print(B[idx])

print(np.abs(A-B))
print(np.abs(B-A))
