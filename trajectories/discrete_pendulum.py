#!/usr/bin/python3

'''
    Трёхсекционный дискретный маятник без управления (u \equiv 0)

    \dot z = f(z, u) => z^{k+1} = z^{k} + \Delta t * f(z^{k}, u^{k})
'''

import numpy as np

from common import Vector
from common import t_start, t_final, z_start, g, l, m, delta_t

from precalculated import fk


import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Алгоритм
N = int(np.ceil((t_final - t_start) / delta_t))


t = np.zeros(N+1)
z = np.zeros((N+1, 6))
t[0] = t_start
z[0] = z_start
for i in range(N):
    t[i+1] = t[i] + delta_t
    z[i+1] = z[i] + fk(z[i], Vector([0.0, 0.0, 0.0]))

# Картинка

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-3, 3))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)


time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    x = [0] + [
        sum([l[k] * np.cos(z[i][k]) for k in range(j+1)]) 
        for j in range(3)
    ]
    y = [0] + [
        sum([l[k] * np.sin(z[i][k]) for k in range(j+1)]) 
        for j in range(3)
    ]

    y_center = [0] + [
        sum([l[k] * np.sin(z[i][k]) for k in range(j)] + np.sin(z[i][j])/2) 
        for j in range(3)
    ]
    Pi = sum([m[i]*g*y_center[i] for i in range(3)])

    line.set_data(x, y)
    time_text.set_text('t = %.2fs P=%.2f' % (t[i], Pi))
    #time_text.set_text(time_template % (res.t[i]))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
                              interval=10, blit=True, init_func=init)


plt.show()