import numpy as np

a = [0.1,0.2,0.7]
b = [0.3,0.3,0.4]
k = []
for i in range(len(a)):
    c = a[i] + b[i]
    k.append(c)
print(k)
l = []
for ii in range(len(k)):
    d = k[ii] / 2
    l.append(d)
print(sum(l))