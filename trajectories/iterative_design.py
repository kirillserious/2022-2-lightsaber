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

z_final = Vector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
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

def picture(z: List[Vector], u: List[Vector], t: List[float])->None:
    # Картинка

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_final, y_final = graphic.pendulum_line(l, z_final)
    ax.plot(x_final, y_final, 'o-', lw=1)
    
    ani = graphic.PendulumAnimation(fig, ax, l, z, t)
    plt.show()

class Cost:
    def __init__(self, model: Model, qf, q, r, step):
        self.__model = model
        self.__qf = qf
        self.__q = q
        self.__r = r
        self.__step = step
    
    def calculate(self, z, u)->float:
        model = self.__model
        qf = self.__qf
        q = self.__q
        r = self.__r
        step = self.__step

        if isinstance(z, list):
            N = len(z)
        else:
            N = z.shape[0]
        
        print('Cost computation')
        cost = sum([
            q(model, z[i], step) + r(model, u[i], step)
            for i in tqdm(range(N-1))
        ])
        cost += qf(model, z[N-1], step)
        return cost


t = np.arange(t_start, t_final, delta_t)

model = Model(l, m, I, g)

#cost = Cost(
#    model,
#    lambda model, z, step: (z-z_final).T.dot(Q_final).dot(z-z_final),
#    lambda model, z, step: step * z.T.dot(Q).dot(z),
#    lambda model, u, step: step * u.T.dot(R).dot(u),
#    0.001,
#)


u_nominal = control.Dummy(model, M, L, z_start, t_start, z_final, t_final, delta_t)
#u_nominal = control.Empty(model, fk, z_start, t_start, t_final, 0.001)
z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)
#costs = [cost.calculate(z_nominal, u_nominal)]
#picture(z_nominal, u_nominal, t)

builder = control.Iterative(
    model,
    fk_z,
    fk_u,
    lambda model, z, step: (z-z_final).T.dot(Q_final).dot(z-z_final),
    lambda model, z, step: Q_final.dot(z-z_final),
    lambda model, z, step: Q_final,
    lambda model, z, step: step * z.T.dot(Q).dot(z),
    lambda model, z, step: step * Q.dot(z),
    lambda model, z, step: step * Q,
    lambda model, u, step: step * u.T.dot(R).dot(u),
    lambda model, u, step: step * R.dot(u),
    lambda model, u, step: step * R,
    0.001,
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
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$y$')
ax.legend(handles=[
    Line2D([0], [0], color='C1', label='Начальная траектория'),
    Line2D([0], [0], color='C0', label='Итерации алгоритма'),
    Line2D([0], [0], color='C3', marker='o', linestyle='None', label='Целевое положение'),
])
ax.view_init(elev=40, azim=-50, roll=0)
plt.savefig('../report/conference/ddp_dummy.pdf', format='pdf', dpi=1200)
exit()
#bx = fig.add_subplot(122)
#bx.plot(list(range(len(costs))), costs)
plt.show()
exit()

while True:
    u = builder.improve(z_nominal, u_nominal)
    z = trajectory(model, fk, z_start, u, 0.001)
    picture(z, u, t)
    z_nominal = z
    u_nominal = u

