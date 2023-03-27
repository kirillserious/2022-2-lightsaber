#!/usr/bin/python3

'''
    Трёхсекционный дискретный маятник без управления (u \equiv 0)

    \dot z = f(z, u) => z^{k+1} = z^{k} + \Delta t * f(z^{k}, u^{k})
'''

import numpy as np

from common import Model, Vector
from common import t_start, t_final, z_start, g, l, m, I, delta_t
import graphic

from precalculated import fk

import matplotlib.pyplot as plt

# Алгоритм
model = Model(l, m, I, g)
N = int(np.ceil((t_final - t_start) / delta_t))

t = np.zeros(N+1)
z = np.zeros((N+1, 6))
t[0] = t_start
z[0] = z_start
for i in range(N):
    t[i+1] = t[i] + delta_t
    z[i+1] = fk(model, z[i], Vector([0.0, 0.0, 0.0]), delta_t)

# Картинка
fig = plt.figure()
ax = fig.add_subplot(111)
ani = graphic.PendulumAnimation(fig, ax, l, z, t)

plt.show()