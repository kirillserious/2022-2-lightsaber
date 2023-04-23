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
z_final = Vector([0.5, 1.1, 1.2, 0.0, 0.0, 0.0])


#speed_target = Vector([0.0, 0.0])
l = [1.2, 1.2, 0.6]
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
else:
    raise Exception('Unexpected start control kind')

z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

def speed_reaching_cost(model: Model, e_target: Vector, z: Vector)->float:
    l = model.l
    return Matrix([[(-np.sin(z[0])*l[0]*z[3] - np.sin(z[1])*l[1]*z[4] - np.sin(z[2])*l[2]*z[5] - e_target[0])**2 + (np.cos(z[0])*l[0]*z[3] + np.cos(z[1])*l[1]*z[4] + np.cos(z[2])*l[2]*z[5] - e_target[1])**2]])

def d_speed_reaching_cost(model: Model, e_target: Vector, z: Vector)-> Vector:
    l = model.l
    return Vector([-2*(-np.sin(z[0])*l[0]*z[3] - np.sin(z[1])*l[1]*z[4] - np.sin(z[2])*l[2]*z[5] - e_target[0])*np.cos(z[0])*l[0]*z[3] - 2*(np.cos(z[0])*l[0]*z[3] + np.cos(z[1])*l[1]*z[4] + np.cos(z[2])*l[2]*z[5] - e_target[1])*np.sin(z[0])*l[0]*z[3], -2*(-np.sin(z[0])*l[0]*z[3] - np.sin(z[1])*l[1]*z[4] - np.sin(z[2])*l[2]*z[5] - e_target[0])*np.cos(z[1])*l[1]*z[4] - 2*(np.cos(z[0])*l[0]*z[3] + np.cos(z[1])*l[1]*z[4] + np.cos(z[2])*l[2]*z[5] - e_target[1])*np.sin(z[1])*l[1]*z[4], -2*(-np.sin(z[0])*l[0]*z[3] - np.sin(z[1])*l[1]*z[4] - np.sin(z[2])*l[2]*z[5] - e_target[0])*np.cos(z[2])*l[2]*z[5] - 2*(np.cos(z[0])*l[0]*z[3] + np.cos(z[1])*l[1]*z[4] + np.cos(z[2])*l[2]*z[5] - e_target[1])*np.sin(z[2])*l[2]*z[5], -2*(-np.sin(z[0])*l[0]*z[3] - np.sin(z[1])*l[1]*z[4] - np.sin(z[2])*l[2]*z[5] - e_target[0])*np.sin(z[0])*l[0] + 2*(np.cos(z[0])*l[0]*z[3] + np.cos(z[1])*l[1]*z[4] + np.cos(z[2])*l[2]*z[5] - e_target[1])*np.cos(z[0])*l[0], -2*(-np.sin(z[0])*l[0]*z[3] - np.sin(z[1])*l[1]*z[4] - np.sin(z[2])*l[2]*z[5] - e_target[0])*np.sin(z[1])*l[1] + 2*(np.cos(z[0])*l[0]*z[3] + np.cos(z[1])*l[1]*z[4] + np.cos(z[2])*l[2]*z[5] - e_target[1])*np.cos(z[1])*l[1], -2*(-np.sin(z[0])*l[0]*z[3] - np.sin(z[1])*l[1]*z[4] - np.sin(z[2])*l[2]*z[5] - e_target[0])*np.sin(z[2])*l[2] + 2*(np.cos(z[0])*l[0]*z[3] + np.cos(z[1])*l[1]*z[4] + np.cos(z[2])*l[2]*z[5] - e_target[1])*np.cos(z[2])*l[2]])

def dd_speed_reaching_cost(model: Model, e_target: Vector, z: Vector)-> Matrix:
    l = model.l
    return Matrix([[-(2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.sin(z[0])*l[0]*z[3] + (-2*np.cos(z[0])*l[0]*z[3] - 2*np.cos(z[1])*l[1]*z[4] - 2*np.cos(z[2])*l[2]*z[5] + 2*e_target[1])*np.cos(z[0])*l[0]*z[3] + 2*np.sin(z[0])**2*l[0]**2*z[3]**2 + 2*np.cos(z[0])**2*l[0]**2*z[3]**2, 2*np.sin(z[0])*np.sin(z[1])*l[0]*l[1]*z[3]*z[4] + 2*np.cos(z[0])*np.cos(z[1])*l[0]*l[1]*z[3]*z[4], 2*np.sin(z[0])*np.sin(z[2])*l[0]*l[2]*z[3]*z[5] + 2*np.cos(z[0])*np.cos(z[2])*l[0]*l[2]*z[3]*z[5], (2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.cos(z[0])*l[0] + (-2*np.cos(z[0])*l[0]*z[3] - 2*np.cos(z[1])*l[1]*z[4] - 2*np.cos(z[2])*l[2]*z[5] + 2*e_target[1])*np.sin(z[0])*l[0], -2*np.sin(z[0])*np.cos(z[1])*l[0]*l[1]*z[3] + 2*np.sin(z[1])*np.cos(z[0])*l[0]*l[1]*z[3], -2*np.sin(z[0])*np.cos(z[2])*l[0]*l[2]*z[3] + 2*np.sin(z[2])*np.cos(z[0])*l[0]*l[2]*z[3]], [2*np.sin(z[0])*np.sin(z[1])*l[0]*l[1]*z[3]*z[4] + 2*np.cos(z[0])*np.cos(z[1])*l[0]*l[1]*z[3]*z[4], -(2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.sin(z[1])*l[1]*z[4] + (-2*np.cos(z[0])*l[0]*z[3] - 2*np.cos(z[1])*l[1]*z[4] - 2*np.cos(z[2])*l[2]*z[5] + 2*e_target[1])*np.cos(z[1])*l[1]*z[4] + 2*np.sin(z[1])**2*l[1]**2*z[4]**2 + 2*np.cos(z[1])**2*l[1]**2*z[4]**2, 2*np.sin(z[1])*np.sin(z[2])*l[1]*l[2]*z[4]*z[5] + 2*np.cos(z[1])*np.cos(z[2])*l[1]*l[2]*z[4]*z[5], 2*np.sin(z[0])*np.cos(z[1])*l[0]*l[1]*z[4] - 2*np.sin(z[1])*np.cos(z[0])*l[0]*l[1]*z[4], (2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.cos(z[1])*l[1] + (-2*np.cos(z[0])*l[0]*z[3] - 2*np.cos(z[1])*l[1]*z[4] - 2*np.cos(z[2])*l[2]*z[5] + 2*e_target[1])*np.sin(z[1])*l[1], -2*np.sin(z[1])*np.cos(z[2])*l[1]*l[2]*z[4] + 2*np.sin(z[2])*np.cos(z[1])*l[1]*l[2]*z[4]], [2*np.sin(z[0])*np.sin(z[2])*l[0]*l[2]*z[3]*z[5] + 2*np.cos(z[0])*np.cos(z[2])*l[0]*l[2]*z[3]*z[5], 2*np.sin(z[1])*np.sin(z[2])*l[1]*l[2]*z[4]*z[5] + 2*np.cos(z[1])*np.cos(z[2])*l[1]*l[2]*z[4]*z[5], -(2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.sin(z[2])*l[2]*z[5] + (-2*np.cos(z[0])*l[0]*z[3] - 2*np.cos(z[1])*l[1]*z[4] - 2*np.cos(z[2])*l[2]*z[5] + 2*e_target[1])*np.cos(z[2])*l[2]*z[5] + 2*np.sin(z[2])**2*l[2]**2*z[5]**2 + 2*np.cos(z[2])**2*l[2]**2*z[5]**2, 2*np.sin(z[0])*np.cos(z[2])*l[0]*l[2]*z[5] - 2*np.sin(z[2])*np.cos(z[0])*l[0]*l[2]*z[5], 2*np.sin(z[1])*np.cos(z[2])*l[1]*l[2]*z[5] - 2*np.sin(z[2])*np.cos(z[1])*l[1]*l[2]*z[5], (2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.cos(z[2])*l[2] + (-2*np.cos(z[0])*l[0]*z[3] - 2*np.cos(z[1])*l[1]*z[4] - 2*np.cos(z[2])*l[2]*z[5] + 2*e_target[1])*np.sin(z[2])*l[2]], [(2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.cos(z[0])*l[0] - (2*np.cos(z[0])*l[0]*z[3] + 2*np.cos(z[1])*l[1]*z[4] + 2*np.cos(z[2])*l[2]*z[5] - 2*e_target[1])*np.sin(z[0])*l[0], 2*np.sin(z[0])*np.cos(z[1])*l[0]*l[1]*z[4] - 2*np.sin(z[1])*np.cos(z[0])*l[0]*l[1]*z[4], 2*np.sin(z[0])*np.cos(z[2])*l[0]*l[2]*z[5] - 2*np.sin(z[2])*np.cos(z[0])*l[0]*l[2]*z[5], 2*np.sin(z[0])**2*l[0]**2 + 2*np.cos(z[0])**2*l[0]**2, 2*np.sin(z[0])*np.sin(z[1])*l[0]*l[1] + 2*np.cos(z[0])*np.cos(z[1])*l[0]*l[1], 2*np.sin(z[0])*np.sin(z[2])*l[0]*l[2] + 2*np.cos(z[0])*np.cos(z[2])*l[0]*l[2]], [-2*np.sin(z[0])*np.cos(z[1])*l[0]*l[1]*z[3] + 2*np.sin(z[1])*np.cos(z[0])*l[0]*l[1]*z[3], (2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.cos(z[1])*l[1] - (2*np.cos(z[0])*l[0]*z[3] + 2*np.cos(z[1])*l[1]*z[4] + 2*np.cos(z[2])*l[2]*z[5] - 2*e_target[1])*np.sin(z[1])*l[1], 2*np.sin(z[1])*np.cos(z[2])*l[1]*l[2]*z[5] - 2*np.sin(z[2])*np.cos(z[1])*l[1]*l[2]*z[5], 2*np.sin(z[0])*np.sin(z[1])*l[0]*l[1] + 2*np.cos(z[0])*np.cos(z[1])*l[0]*l[1], 2*np.sin(z[1])**2*l[1]**2 + 2*np.cos(z[1])**2*l[1]**2, 2*np.sin(z[1])*np.sin(z[2])*l[1]*l[2] + 2*np.cos(z[1])*np.cos(z[2])*l[1]*l[2]], [-2*np.sin(z[0])*np.cos(z[2])*l[0]*l[2]*z[3] + 2*np.sin(z[2])*np.cos(z[0])*l[0]*l[2]*z[3], -2*np.sin(z[1])*np.cos(z[2])*l[1]*l[2]*z[4] + 2*np.sin(z[2])*np.cos(z[1])*l[1]*l[2]*z[4], (2*np.sin(z[0])*l[0]*z[3] + 2*np.sin(z[1])*l[1]*z[4] + 2*np.sin(z[2])*l[2]*z[5] + 2*e_target[0])*np.cos(z[2])*l[2] - (2*np.cos(z[0])*l[0]*z[3] + 2*np.cos(z[1])*l[1]*z[4] + 2*np.cos(z[2])*l[2]*z[5] - 2*e_target[1])*np.sin(z[2])*l[2], 2*np.sin(z[0])*np.sin(z[2])*l[0]*l[2] + 2*np.cos(z[0])*np.cos(z[2])*l[0]*l[2], 2*np.sin(z[1])*np.sin(z[2])*l[1]*l[2] + 2*np.cos(z[1])*np.cos(z[2])*l[1]*l[2], 2*np.sin(z[2])**2*l[2]**2 + 2*np.cos(z[2])**2*l[2]**2]])


builder = control.Iterative1(
    f_z = lambda z, u, step: fk_z(model, z, u, step),
    f_u = lambda z, u, step: fk_u(model, z, u, step),
    qf = cost.Reaching(model, e_target),
    q = cost.Empty(),
    r = delta_t * 0.0000001 * cost.Energy(),
    step = delta_t,
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