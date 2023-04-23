#!/usr/bin/python3

'''
    Трёхсекционный дискретный маятник без управления (u \equiv 0)

    \dot z = f(z, u) => z^{k+1} = z^{k} + \Delta t * f(z^{k}, u^{k})
'''
import os
import numpy as np

from common import Model, Vector, trajectory, end_effector
from common import t_start, t_final, z_start, g, l, m, I, delta_t
import graphic

from precalculated import fk

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Входные данные
l = [1.2, 1.2, 0.6]
t_start = 0.0
t_final = 1.0
z_start = Vector([0, 0, 0, 0, 0, 0])

# Алгоритм
model = Model(l, m, I, g)

step = delta_t
t = np.arange(t_start, t_final, step)
u = np.zeros((t.shape[0]-1, 3))
z = trajectory(model, fk, z_start, u, step)

# Картинка
fig = plt.figure()
ax = fig.add_subplot(111)

graph = os.getenv('GRAPH', default='animation')
if graph == 'animation':
    ani = graphic.PendulumAnimation(fig, ax, l, z, t)
    plt.show()
elif graph == 'pendulum':
    graphic.add_pendulum_lines(ax, l, z, lines=10)

    end_effectors = [[], []]
    for i in range(len(z)):
        ef = end_effector(z[i], l)
        end_effectors[0] += [ef[0]]
        end_effectors[1] += [ef[1]]

    ax.plot(end_effectors[0], end_effectors[1], c='C1')
    ax.legend(handles=[
        Line2D([0], [0], color='C0', label='Сочленения руки'),
        Line2D([0], [0], color='C1', label='Траектория схвата')
    ])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    output = os.getenv('OUTPUT')
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format='pdf', dpi=1200)
else:
    raise Exception('Unexpected graph kind')