
import numpy as np

a = np.array([[0,0,0]])
b = np.array([3,4])
d = np.array([[4,5,6]])
print(a[-1])
c = np.hstack((b, np.array([0])))[np.newaxis, :]
print(c[-1])

c = np.append(c, d, axis=0)
print(c)

