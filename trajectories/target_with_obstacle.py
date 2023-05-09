#!/usr/bin/python3
'''
    START = empty | dummy
    ITER = 5 
    GRAPH = endpoint | pendulum
'''


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
import os
from typing import List
from common import Vector, Matrix, Model, trajectory, end_effector
from common import t_start, z_start, t_final, l, m, I, g, delta_t
import graphic
import control
import cost

from precalculated import fk, fk_u, fk_z, M, L

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Входные даннные

#z_final = Vector([0.8, 1.1, 1.2, 0.0, 0.0, 0.0])
#l = [1.2, 1.2, 0.6]

z_final = Vector([-0.5, 1.1, 1.4, 0, 0, 0])
l = [0.7, 0.7, 1.6]
m = [0.08, 0.08, 0.1]
I = [m[i] * l[i] * l[i] / 3 for i in range(3)]

# Алгоритм
delta_t = 0.001
N = int(np.ceil((t_final - t_start) / delta_t))


t = np.arange(t_start, t_final, delta_t)

model = Model(l, m, I, g)

start = os.getenv('START', default='empty')
if start == 'empty':
    u_nominal = control.Empty(model, fk, z_start, t_start, t_final, delta_t)
elif start == 'lqr':
    u_nominal = control.LQR(model, M, L, z_start, t_start, z_final, t_final, delta_t)
elif start == 'dummy':
    u_nominal = control.Dummy(model, M, L, z_start, t_start, z_final, t_final, delta_t)
else:
    raise Exception('Unexpected start control kind')

z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

r_obstacle = 0.1
obstacle = Vector([2.2, -2.0])

step = 0.001
builder = control.Iterative1(
    f_z = lambda z, u, step: fk_z(model, z, u, step),
    f_u = lambda z, u, step: fk_u(model, z, u, step),
    qf = 100.0 * cost.TargetFinal(z_final),
    q = step * 0.2 * cost.Obstacle(model, obstacle, r_obstacle),
    r = step *  cost.Energy(),
    step = step,
    #max_d=500.0,
)


graph = os.getenv('GRAPH', default='endpoint')
output = os.getenv('OUTPUT')
iterations = int(os.getenv('ITER', default='5'))

if graph == 'endpoint':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    graphic.end_effector_lines(ax, l, [z_nominal], t_start, 0.001, color='C1')
    zs = []
    for i in range(iterations):
        print('%d iteration started' % (i+1))
        u = builder.improve(z_nominal, u_nominal)
        z = trajectory(model, fk, z_start, u, 0.001)
        zs = zs + [z]
        z_nominal = z
        u_nominal = u

    graphic.end_effector_lines(ax, l, zs, t_start, 0.001)
    target = end_effector(z_final, l)
    ax.plot3D([t_final], [target[0]], [target[1]], 'o-', lw=2, c='C2')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$y$')
    ax.legend(handles=[
        Line2D([0], [0], color='C1', label='Начальная траектория'),
        Line2D([0], [0], color='C0', label='Итерации алгоритма'),
        Line2D([0], [0], color='C2', marker='o', linestyle='None', label='Целевое положение'),
    ])
    ax.view_init(elev=40, azim=-50, roll=0)
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format='pdf', bbox_inches='tight', dpi=1200)
    exit()

if graph == 'pendulum':
    for i in range(iterations):
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
    circle = Circle((obstacle[0], obstacle[1]), r_obstacle, color='C3')
    ax.add_patch(circle)
    ax.legend(handles=[
        Line2D([0], [0], color='C0', label='Сочленения руки'),
        Line2D([0], [0], color='C1', label='Траектория схвата'),
        Circle((0,0), 0, color='C3', label='Препятствие')
    ])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format='pdf', bbox_inches='tight', dpi=1200)
    exit()

raise Exception('Unexpected graph kind')