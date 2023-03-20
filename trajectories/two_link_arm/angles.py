#!/usr/bin/python3

import numpy as np
Vector = np.array

# Параметры модели
g = 9.8
m = [0.1, 0.1]
l = [1, 1]
I = [m[i] * l[i] * l[i] / 3 for i in range(2)]

# Входные параметры
position = Vector([1.0, 1.0])

a = (-l[1]*position[1] + position[1]*(l[0] + l[1]) + np.sqrt(l[0]*l[1]*(l[0] + l[1] - position[1])*(l[0] + l[1] + position[1])))/(l[0]*(l[0] + l[1]))
b = (l[1]*position[1] - np.sqrt(l[0]*l[1]*(l[0]**2 + 2*l[0]*l[1] + l[1]**2 - position[1]**2)))/(l[1]*(l[0] + l[1]))
print('%f %f' % (a, b))

a = (l[0]*position[1] - np.sqrt(l[0]*l[1]*(l[0]**2 + 2*l[0]*l[1] + l[1]**2 - position[1]**2)))/(l[0]*(l[0] + l[1]))
b = (l[1]*position[1] + np.sqrt(l[0]*l[1]*(l[0] + l[1] - position[1])*(l[0] + l[1] + position[1])))/(l[1]*(l[0] + l[1]))
print('%f %f' % (a, b))

exit()
roots = np.roots([
    l[1]**2/l[0] + l[1],
    - l[1] / l[0] * position[1],
    - l[0] + 1/l[0]*position[1]**2 - l[1] + position[0], 
])
print(roots)
for root in roots:
    b = np.arcsin(root)
    a = np.arcsin((position[1] - l[1]*root)/l[0])
    print('Angles %f %f' % (a, b))