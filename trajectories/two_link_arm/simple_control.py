#!/usr/bin/python3

import numpy as np
from tqdm import tqdm
from common import Model
import graphic
import control
Vector = np.array
Matrix = np.array

# Входные параметры

# Параметры модели
g = 9.8
m = [0.1, 0.1]
l = [1, 1]
I = [m[i] * l[i] * l[i] / 3 for i in range(2)]

# Параметры задачи управления
t_start = 0.0
t_final = 1.0
z_start = Vector([0.0, 0.0, 0.0, 0.0])
z_final = Vector([1.57, 0.0, 0.0, 0.0])

# Параметры дискретизации
delta_t = 0.001
N = int(np.ceil((t_final - t_start) / delta_t))

t = np.zeros(N+1)
t[0] = t_start
for i in range(0, N):
    t[i+1] = t[i] + delta_t


z, u = control.Dummy(
    Model(l, m, I, g),
    None, None,
    z_start, t_start,
    z_final, t_final,
    delta_t,
)

# Картинка
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ani = graphic.PendulumAnimation(fig, ax, l, z, t)
plt.show()
