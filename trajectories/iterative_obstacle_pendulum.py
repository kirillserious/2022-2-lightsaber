#!/usr/bin/python3

'''
    Трёхсекционный дискретный маятник
    
        z^{k+1} = z^{k} + \Delta t * f(z^{k}, u^{k})

    c управлением u, минимизирующем

        J = \langle x - x^{N}, Q_N (x - x^{N}) \\rangle + \sum_{k=0}^{N-1} ( \langle x^{k}, Q x^{k} \\rangle + \langle u^{k}, R u^{k} \\rangle) 

    let u_nominal == 0, x_nominal - corresponding trajectory:
        u = u_nominal + delta_u
        x = x_nominal + delta_x
    
    delta_x^{k+1} = A^{k} * delta_x^{k} + B^{k} * delta_u^{k}

'''
from typing import List
from common import Vector, Matrix, Model, trajectory, end_effector
from common import t_start, z_start, t_final, l, m, I, g, delta_t
import graphic
import control

from precalculated import fk, fk_u, fk_z, M, L

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm


# Входные даннные

z_final = Vector([0.8, 1.1, 1.2, 0.0, 0.0, 0.0])
l = [1.2, 1.2, 0.6]
Q_final = 1000000.0 *Matrix([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])
Q = Matrix([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])
R = Matrix([
    [0.000001, 0.0, 0.0],
    [0.0, 0.000001, 0.0],
    [0.0, 0.0, 0.000001],
])

# Алгоритм
delta_t = 0.001
N = int(np.ceil((t_final - t_start) / delta_t))


t = np.arange(t_start, t_final, delta_t)

model = Model(l, m, I, g)


u_nominal = control.Dummy(model, M, L, z_start, t_start, z_final, t_final, delta_t)
z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

import numpy

r_obstacle = 0.1
def normal_dist(q, obstacle):
    return (-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**(-2.0)
def d_normal_dist(q, obstacle):
    return numpy.array([[(-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[0])*l[0] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[0])*l[0])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[1])*l[1] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[1])*l[1])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[2])*l[2] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[2])*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, 0, 0, 0]])

def dd_normal_dist(q, obstacle):
    return numpy.array([[(-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[0])*l[0] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[0])*l[0])*(-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[0])*l[0] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[0])*l[0])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-(-4*numpy.sin(q[0])*l[0] - 4*numpy.sin(q[1])*l[1] - 4*numpy.sin(q[2])*l[2] + 4*obstacle[1])*numpy.sin(q[0])*l[0] + (4*numpy.cos(q[0])*l[0] + 4*numpy.cos(q[1])*l[1] + 4*numpy.cos(q[2])*l[2] - 4*obstacle[0])*numpy.cos(q[0])*l[0] - 4*numpy.sin(q[0])**2*l[0]**2 - 4*numpy.cos(q[0])**2*l[0]**2)/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[0])*l[0] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[0])*l[0])*(-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[1])*l[1] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[1])*l[1])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-4*numpy.sin(q[0])*numpy.sin(q[1])*l[0]*l[1] - 4*numpy.cos(q[0])*numpy.cos(q[1])*l[0]*l[1])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[0])*l[0] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[0])*l[0])*(-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[2])*l[2] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[2])*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-4*numpy.sin(q[0])*numpy.sin(q[2])*l[0]*l[2] - 4*numpy.cos(q[0])*numpy.cos(q[2])*l[0]*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, 0, 0, 0], [(-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[0])*l[0] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[0])*l[0])*(-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[1])*l[1] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[1])*l[1])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-4*numpy.sin(q[0])*numpy.sin(q[1])*l[0]*l[1] - 4*numpy.cos(q[0])*numpy.cos(q[1])*l[0]*l[1])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[1])*l[1] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[1])*l[1])*(-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[1])*l[1] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[1])*l[1])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-(-4*numpy.sin(q[0])*l[0] - 4*numpy.sin(q[1])*l[1] - 4*numpy.sin(q[2])*l[2] + 4*obstacle[1])*numpy.sin(q[1])*l[1] + (4*numpy.cos(q[0])*l[0] + 4*numpy.cos(q[1])*l[1] + 4*numpy.cos(q[2])*l[2] - 4*obstacle[0])*numpy.cos(q[1])*l[1] - 4*numpy.sin(q[1])**2*l[1]**2 - 4*numpy.cos(q[1])**2*l[1]**2)/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[1])*l[1] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[1])*l[1])*(-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[2])*l[2] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[2])*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-4*numpy.sin(q[1])*numpy.sin(q[2])*l[1]*l[2] - 4*numpy.cos(q[1])*numpy.cos(q[2])*l[1]*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, 0, 0, 0], [(-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[0])*l[0] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[0])*l[0])*(-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[2])*l[2] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[2])*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-4*numpy.sin(q[0])*numpy.sin(q[2])*l[0]*l[2] - 4*numpy.cos(q[0])*numpy.cos(q[2])*l[0]*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[1])*l[1] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[1])*l[1])*(-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[2])*l[2] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[2])*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-4*numpy.sin(q[1])*numpy.sin(q[2])*l[1]*l[2] - 4*numpy.cos(q[1])*numpy.cos(q[2])*l[1]*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, (-6*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[2])*l[2] + 6*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[2])*l[2])*(-4*(numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])*numpy.cos(q[2])*l[2] + 4*(numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])*numpy.sin(q[2])*l[2])/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**4 + (-(-4*numpy.sin(q[0])*l[0] - 4*numpy.sin(q[1])*l[1] - 4*numpy.sin(q[2])*l[2] + 4*obstacle[1])*numpy.sin(q[2])*l[2] + (4*numpy.cos(q[0])*l[0] + 4*numpy.cos(q[1])*l[1] + 4*numpy.cos(q[2])*l[2] - 4*obstacle[0])*numpy.cos(q[2])*l[2] - 4*numpy.sin(q[2])**2*l[2]**2 - 4*numpy.cos(q[2])**2*l[2]**2)/(-r_obstacle + (numpy.sin(q[0])*l[0] + numpy.sin(q[1])*l[1] + numpy.sin(q[2])*l[2] - obstacle[1])**2 + (numpy.cos(q[0])*l[0] + numpy.cos(q[1])*l[1] + numpy.cos(q[2])*l[2] - obstacle[0])**2)**3, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

obstacle = Vector([2.0, -2.0])

builder = control.Iterative(
    model,
    fk_z,
    fk_u,
    lambda model, z, step: (z-z_final).T.dot(Q_final).dot(z-z_final),
    lambda model, z, step: Q_final.dot(z-z_final),
    lambda model, z, step: Q_final,
    lambda model, z, step: step * z.T.dot(Q).dot(z) + step * normal_dist(z, obstacle),
    lambda model, z, step: step * Q.dot(z) + step * d_normal_dist(z, obstacle),
    lambda model, z, step: step * Q + step * dd_normal_dist(z, obstacle),
    lambda model, u, step: step * u.T.dot(R).dot(u),
    lambda model, u, step: step * R.dot(u),
    lambda model, u, step: step * R,
    0.001,
)

for i in range(10):
    print('%d iteration started' % (i+1))
    u = builder.improve(z_nominal, u_nominal)
    z = trajectory(model, fk, z_start, u, 0.001)
    z_nominal = z
    u_nominal = u

fig = plt.figure()
ax = fig.add_subplot(111)
graphic.add_pendulum_lines(ax, l, z_nominal, lines=10)

end_effectors = [[], []]
for i in range(len(z_nominal)):
    ef = end_effector(z_nominal[i], l)
    end_effectors[0] += [ef[0]]
    end_effectors[1] += [ef[1]]

ax.plot(end_effectors[0], end_effectors[1], c='C1')

from matplotlib.patches import Circle
circle = Circle((2.0, -2.0), 0.2, color='C3')
ax.add_patch(circle)
ax.legend(handles=[
    Line2D([0], [0], color='C0', label='Сочленения руки'),
    Line2D([0], [0], color='C1', label='Траектория схвата'),
    Circle((0,0), 0, color='C3', label='Препятствие')
])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.savefig('../report/conference/obstacle_pendulum.pdf', format='pdf', dpi=1200)
#plt.show()