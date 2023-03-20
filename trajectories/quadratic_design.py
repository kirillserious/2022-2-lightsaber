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
from typing import List, Tuple
from common import Vector, Matrix
from common import t_start, z_start, t_final, l, m, I, g, delta_t

from precalculated import fk, fk_u, fk_z

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#import scipy.integrate


# Входные даннные

z_final = Vector([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
Q_final = 10000.0 *Matrix([
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


def improve(z_nominal: List[Vector], u_nominal: List[Vector])-> Tuple[List[Vector], List[Vector]]:
    print('Calculation of S started')

    K = [None] * N
    K_u = [None] * N
    K_v = [None] * N
    v = [None] * N
    v[N-1] = np.matmul(Q_final, z_nominal[N-1] - z_final)

    Si = Q_final
    #print(np.linalg.norm(Si))
    #exit()
    for i in tqdm(range(N-2, -1, -1)):
        ui = u_nominal[i]
        zi = z_nominal[i]

        Ai = fk_z(zi, ui)
        #print(np.linalg.norm(Ai))
        #print()
        Bi = fk_u(zi, ui)
        Bi_T = np.transpose(Bi)
        Bi_T_Si = Bi_T.dot(Si)
        support = np.linalg.inv(Bi_T_Si.dot(Bi) + R)


        K[i] = support.dot(Bi_T_Si).dot(Ai)
        K_v[i] = support.dot(Bi_T)
        K_u[i] = support.dot(R)

        Ai_minus_Bi_K = Ai - Bi.dot(K[i])

        Si = np.transpose(Ai).dot(Si).dot(Ai_minus_Bi_K) + Q
        v[i] = np.matmul(np.transpose(Ai_minus_Bi_K), v[i+1]) - np.matmul(np.transpose(K[i]).dot(R), ui) + np.matmul(Q, zi)
        #print(np.linalg.norm(Si))
        #print(np.linalg.norm(K_v[i]))
        #print(np.linalg.norm(K_u[i]))
        #print(np.linalg.norm(v[i]))
        #print()
        
    print('Calculation of control started')

    z = [None] * N
    u = [None] * N
    t = [None] * N
    t[0] = t_start
    delta_zi = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in tqdm(range(N)):
        if i < N-1:
            delta_ui = -np.matmul(K[i], delta_zi) - np.matmul(K_v[i], v[i+1]) - np.matmul(K_u[i], u_nominal[i])

        #print(np.linalg.norm(delta_ui))
        #print()
        u[i] = u_nominal[i] + delta_ui
        #print(u_nominal[i])
        #print(delta_ui)
        #print()

        Ai = fk_z(z_nominal[i], u_nominal[i])
        Bi =  fk_u(z_nominal[i], u_nominal[i])
        delta_zi = np.matmul(Ai, delta_zi) + np.matmul(Bi, delta_ui)
        z[i] = z_nominal[i] + delta_zi
        if i > 0:
            t[i] = t[i-1] + delta_t

    print('Calculation of cost function')


    return z, u

def picture(z: List[Vector], u: List[Vector], t: List[float])->None:
    # Картинка

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-3, 3))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)

    x_final = [0] + [
            sum([l[k] * np.cos(z_final[k]) for k in range(j+1)]) 
            for j in range(3)
        ]
    y_final = [0] + [
            sum([l[k] * np.sin(z_final[k]) for k in range(j+1)]) 
            for j in range(3)
        ]
    line_final, = ax.plot(x_final, y_final, 'o-', lw=1)


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

        u_norm = sum([u[i][j]**2 for j in range(3)])

        line.set_data(x, y)
        time_text.set_text('t = %.2fs |u|=%.2f' % (t[i], u_norm))
        #time_text.set_text(time_template % (res.t[i]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)-1),
                                interval=1, blit=True, init_func=init)


    plt.show()


print('Initialize nominal control')
u_nominal = [Vector([0.0, 0.0, 0.0]) for i in tqdm(range(N))]
for i in range(int(N/6*4), int(N/6*5)):
    u_nominal[i] = Vector([10.0, -10.0, -10.0])
for i in range(int(N/6*5), N):
    u_nominal[i] = Vector([-10.0, 10.0, 10.0])

print('Initialize nominal trajectory')
z_nominal = [None] * N
z_nominal[0] = z_start
for i in tqdm(range(N-1)):
    z_nominal[i+1] = fk(z_nominal[i], u_nominal[i])



t = [None] * N
t[0] = t_start
for i in range(N-1):
    t[i+1] = t[i] + delta_t

picture(z_nominal, u_nominal, t)

while True:
    z, u = improve(z_nominal, u_nominal)
    picture(z, u, t)
    z_nominal = z
    u_nominal = u