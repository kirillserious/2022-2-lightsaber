#!/usr/bin/python3
from typing import Tuple

import numpy as np
Vector = np.array
Matrix = np.array

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.integrate

# Classic case

# Дано:
m0 = 1
m1 = 1
m2 = 1

l0 = 1
l1 = 1
l2 = 1

I0 = m0 * l0 * l0 / 12
I1 = m1 * l1 * l1 / 12
I2 = m2 * l2 * l2 / 12



# Предпосчитано:
def M(theta: Vector) -> Matrix:
    return Matrix([
        [I0 + m0*(l0**2*sin(q0(t))**2/2, ]
    ])

def C(theta: Vector, dtheta: Vector) -> Matrix:
    return Matrix([
        [-np.sin(theta[1]) * dtheta[1], -np.sin(theta[1]) * dtheta[1] - -np.sin(theta[1]) * dtheta[0]],
        [np.cos(theta[0] + theta[1]), 0]
    ])

def g(theta: Vector) -> Vector:
    return Vector([
        np.cos(theta[0]) + np.cos(theta[0] + theta[1]),
        np.cos(theta[0] + theta[1])
    ])

def x2theta(x: Vector) -> Tuple[Vector, Vector]:
    theta = x[:2]
    dtheta = x[2:]
    return (theta, dtheta)

def theta2x(theta: Vector, dtheta: Vector) -> Vector:
    return np.concatenate((theta, dtheta))


def func(t: float, x: Vector)->Vector:
    
    theta, dtheta = x2theta(x)
    
    ddtheta = np.matmul(
        np.linalg.inv(M(theta)),
        ( - np.matmul(C(theta, dtheta), dtheta) - g(theta))
    )

    return np.concatenate((dtheta, ddtheta))

res = scipy.integrate.solve_ivp(func, (0, 5), Vector([0, 0, 0, 0]))
print(res.y[0])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 5), ylim=(-2, 5))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)


time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    x1 = np.cos(res.y[0][i]) * l1
    y1 = np.sin(res.y[0][i]) * l1
    x2 = x1 + l2 * np.cos(res.y[1][i] + res.y[0][i])
    y2 = y1 + l2 * np.sin(res.y[1][i] + res.y[0][i])
    #thisx = [0, x1[i], x2[i]]
    #thisy = [0, y1[i], y2[i]]

    line.set_data([0, x1, x2], [0, y1, y2])
    time_text.set_text(time_template % (res.t[i]))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(res.t)),
                              interval=50, blit=True, init_func=init)

#plt.plot(res.t, res.y[0])
plt.show()
#print(res.y[0])