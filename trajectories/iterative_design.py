#!/usr/bin/python3

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
from common import Vector, Matrix, Model, trajectory
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

u_nominal = control.Dummy(model, M, L, z_start, t_start, z_final, t_final, delta_t)
z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

picture(z_nominal, u_nominal, t)

builder = control.Iterative(
    model,
    fk_z,
    fk_u,
    lambda model, z, step: (z-z_final).T.dot(Q_final).dot(z-z_final),
    lambda model, z, step: Q_final.dot(z-z_final),
    lambda model, z, step: Q_final,
    lambda model, z, step: z.T.dot(Q).dot(z),
    lambda model, z, step: Q.dot(z),
    lambda model, z, step: Q,
    lambda model, u, step: u.T.dot(R).dot(u),
    lambda model, u, step: R.dot(u),
    lambda model, u, step: R,
    0.001,
)

while True:
    u = builder.improve(z_nominal, u_nominal)
    z = trajectory(model, fk, z_start, u, 0.001)
    picture(z, u, t)
    z_nominal = z
    u_nominal = u

#builder = control.QuadraticIterative(
#    model=model, 
#    fk_z=fk_z, 
#    fk_u=fk_u,
#    z_final=z_final,
#    Q_final=Q_final,
#    Q=Q,
#    R=R
#)



#while True:
#    z, u = builder.improve(z_nominal, u_nominal)
#    picture(z, u, t)
#    z_nominal = z
#    u_nominal = u