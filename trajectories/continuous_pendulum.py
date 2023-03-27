#!/usr/bin/python3

'''
    Непрерывный трёхсекционный маятник без потери энергии без управления

    \ddot q = - M^{-1}(q) L(q, \dot q) 
'''

from typing import Tuple

import numpy as np
from common import Model, Vector, Matrix
from common import t_start, t_final, z_start, g, l, m, I
from precalculated import M_inv, L
import graphic

import matplotlib.pyplot as plt


import scipy.integrate

model = Model(
    l, m, I, g,
)

def z2theta(z: Vector) -> Tuple[Vector, Vector]:
    theta = z[:3]
    dtheta = z[3:]
    return (theta, dtheta)

def theta2z(theta: Vector, dtheta: Vector) -> Vector:
    return np.concatenate((theta, dtheta))


def func(t: float, z: Vector)->Vector:
    theta, dtheta = z2theta(z)
    ddtheta = np.matmul(M_inv(model, theta), Vector([0.0, 0.0, 0.0])) +  np.matmul(M_inv(model, theta), -L(model, z))
    return np.concatenate((dtheta, ddtheta))

res = scipy.integrate.solve_ivp(func, (t_start, t_final), z_start, max_step=0.01)
if not res.success:
    print('ODE solver failed')
    print(res.message)
    exit(1)

fig = plt.figure()
ax = fig.add_subplot(111)
ani = graphic.PendulumAnimation(fig, ax, l, res.y.T, res.t)

plt.show()