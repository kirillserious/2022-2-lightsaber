#!/usr/bin/python3

'''
    Трёхсекционный дискретный маятник без управления (u \equiv 0)

    \dot z = f(z, u) => z^{k+1} = z^{k} + \Delta t * f(z^{k}, u^{k})
'''

import numpy as np

from common import Model, Vector, trajectory
from common import t_start, t_final, z_start, g, l, m, I, delta_t
import graphic

from precalculated import fk

import matplotlib.pyplot as plt

# Алгоритм
model = Model(l, m, I, g)

step = delta_t
t = np.arange(t_start, t_final, step)
u = np.zeros((t.shape[0]-1, 3))
z = trajectory(model, fk, z_start, u, step)

# Картинка
fig = plt.figure()
ax = fig.add_subplot(111)
ani = graphic.PendulumAnimation(fig, ax, l, z, t)

plt.show()