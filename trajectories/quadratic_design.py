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


# Входные даннные

z_final = Vector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
Q_final = 100000.0 *Matrix([
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

def picture(z: List[Vector], u: List[Vector], t: List[float])->None:
    # Картинка

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_final, y_final = graphic.pendulum_line(l, z_final)
    ax.plot(x_final, y_final, 'o-', lw=1)
    
    ani = graphic.PendulumAnimation(fig, ax, l, z, t)
    plt.show()


t = np.arange(t_start, t_final, delta_t)

model = Model(l, m, I, g)

if True:
    u_nominal = control.Empty(
        model, fk, z_start, t_start, t_final, delta_t,
    )
    z_nominal = trajectory(
        model,
        fk,
        z_start,
        u_nominal,
        delta_t,
    )
else:
    u_nominal = control.Dummy(
        model,
        M, L,
        z_start, t_start,
        z_final, t_final,
        delta_t, 
    )
    z_nominal = trajectory(
        model,
        fk,
        z_start,
        u_nominal,
        delta_t,
    )

picture(z_nominal, u_nominal, t)



builder = control.QuadraticIterative(
    model=model, 
    fk_z=fk_z, 
    fk_u=fk_u,
    z_final=z_final,
    Q_final=Q_final,
    Q=Q,
    R=R
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
graphic.end_effector_lines(ax, l, [z_nominal], t_start, 0.001, color='C1')

zs = []
for i in range(10):
    print('%d iteration started' % (i+1))
    u = builder.improve(z_nominal, u_nominal)
    z = trajectory(model, fk, z_start, u, 0.001)
    zs = zs + [z]
    #costs = costs + [cost.calculate(z, u)]
    z_nominal = z
    u_nominal = u

graphic.end_effector_lines(ax, l, zs, t_start, 0.001)
target = end_effector(z_final, l)
ax.plot3D([t_final], [target[0]], [target[1]], 'o-', lw=2, c='C3')

#bx = fig.add_subplot(122)
#bx.plot(list(range(len(costs))), costs)
plt.show()
exit()

while True:
    u = builder.improve(z_nominal, u_nominal)
    z = trajectory(model, fk, z_start, u, delta_t)
    picture(z, u, t)
    z_nominal = z
    u_nominal = u