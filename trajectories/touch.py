#!/usr/bin/python3

import os
from typing import List
from common import Vector, Matrix, Model, trajectory, end_effector
from common import t_start, z_start, t_final, m, I, g, delta_t
from precalculated import fk, fk_u, fk_z, M, L
import graphic
import control
import cost

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

# Входные даннные
z_final = Vector([-1.2, -0.9, 0.8, 0.0, 0.0, 4.0])


#speed_target = Vector([0.0, 0.0])
l = [0.7, 0.7, 1.6]
m = [0.08, 0.08, 0.1]
I = [m[i] * l[i] * l[i] / 3 for i in range(3)]
e_target = Vector([
    sum([l[i]*np.cos(z_final[i]) for i in range(3)]),
    sum([l[i]*np.sin(z_final[i]) for i in range(3)]),
])


# Алгоритм
delta_t = 0.001
N = int(np.ceil((t_final - t_start) / delta_t))
t = np.arange(t_start, t_final, delta_t)

model = Model(l, m, I, g)

start = os.getenv('START', default='empty')
if start == 'empty':
    u_nominal = control.Empty(model, fk, z_start, t_start, t_final, delta_t)
elif start == 'dummy':
    #raise Exception('Unexpected start control kind')
    u_nominal = control.Dummy(model, M, L, z_start, t_start, z_final, t_final, delta_t)
elif start == 'lqr':
    #raise Exception('Unexpected start control kind')
    u_nominal = control.LQR(model, M, L, z_start, t_start, z_final, t_final, delta_t)
else:
    raise Exception('Unexpected start control kind')

z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

builder = control.Iterative3(
    f_z = lambda z, u, step: fk_z(model, z, u, step),
    f_u = lambda z, u, step: fk_u(model, z, u, step),
    #qf = cost.Reaching(model, e_target),
    qf = 100.0 * cost.Touch(model, e_target) + cost.ReachingSpeed(model, Vector([1.0, 0.0])),
    q = delta_t  * cost.DummyPhase(),
    r = delta_t  * cost.Energy(),
    step = delta_t,
    max_d=10000000000.0,
)

graph = os.getenv('GRAPH', default='animation')
if graph == 'animation':
    def picture(z: List[Vector], u: List[Vector], t: List[float])->None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(e_target[0], e_target[1], 'o-', lw=1)
        #x_final, y_final = graphic.pendulum_line(l, z_final)
        #ax.plot(x_final, y_final, 'o-', lw=1)
        
        ani = graphic.PendulumAnimation(fig, ax, l, z, t)
        plt.show()

    picture(z_nominal, u_nominal, t)
    while True:
        u = builder.improve(z_nominal, u_nominal)
        z = trajectory(model, fk, z_start, u, 0.001)
        picture(z, u, t)
        z_nominal = z
        u_nominal = u
    exit()

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
        z = trajectory(model, fk, z_start, u, delta_t)
        zs = zs + [z]

        z_nominal = z
        u_nominal = u

    graphic.end_effector_lines(ax, l, zs, t_start, delta_t)
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
        z = trajectory(model, fk, z_start, u, delta_t)
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
    ax.plot([e_target[0]], [e_target[1]], 'o-', lw=2, c='C2')
    ax.legend(handles=[
        Line2D([0], [0], color='C0', label='Сочленения руки'),
        Line2D([0], [0], color='C1', label='Траектория схвата'),
        Line2D([0], [0], marker='o', linestyle='None', color='C2', label='Положение мяча')
    ])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format='pdf', bbox_inches='tight', dpi=1200)
    exit()