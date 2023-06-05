#!/usr/bin/python3
'''
    START = empty | dummy
    ITER = 5 
    GRAPH = animation | endpoint | pendulum
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
from tqdm import tqdm

if False:
    lqr_cost = [2.827790747247821, 0.8019510559005684, 0.45734151090075237, 0.4560181402567976]
    empty_cost = [70.37746380020212, 58.59558697629737, 47.91318130772849, 38.29744369196915, 29.779676332919372, 22.397019743585787, 16.143297225204737, 10.966151791412665, 6.802060919316443, 3.6236686101711326, 1.4740907129469702, 0.49028857749956295, 0.4570647599846641, 0.456019484491975]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(list(range(1, len(empty_cost)+1)), [c - empty_cost[len(empty_cost) - 1] for c in empty_cost], 'o-')
    ax.plot(list(range(1, len(lqr_cost)+1)), [c - lqr_cost[len(lqr_cost) - 1] for c in lqr_cost], 'o-')
    ax.set_yscale('log')
    ax.grid(True)
    ax.set_xlim([1, 14])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Итерация алгоритма')
    ax.set_ylabel('$J - J*$')
    ax.legend(["Нулевое реф. управление", "Лин.-квадр. реф. управление"])
    plt.savefig('compare.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    #plt.show()

    exit()
# Входные даннные

z_final = Vector([-0.5, 1.1, 1.4, -5.0, -5.0, -5.0])
l = [0.7, 0.7, 1.6]

Q_final = 1000000.0 *Matrix([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
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
    u_nominal = control.Dummy(model, M, L, z_start, t_start, z_final, t_final, delta_t)
elif start == 'lqr':
    u_nominal = control.LQR(model, M, L, z_start, t_start, z_final, t_final, delta_t)
else:
    raise Exception('Unexpected start control kind')

z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

if True:
    builder = control.Iterative3(
        f_z = lambda z, u, step: fk_z(model, z, u, step),
        f_u = lambda z, u, step: fk_u(model, z, u, step),
        qf = 100.0 * cost.TargetFinal(z_final),
        q = delta_t * cost.DummyPhase(),
        r = delta_t *  cost.Energy(),
        step = delta_t,
        max_d=100.0,
    )

def full_cost(u, z)->float:
    if isinstance(z, list):
        N = len(z)
    else:
        N = z.shape[0]

    result = 0
    for k in range(N-1):
        result += delta_t * cost.Energy().cost(u[k], delta_t)
        result += delta_t * cost.DummyPhase().cost(z[k], delta_t)
    result += 100.0 * cost.TargetFinal(z_final).cost(z[N-1], delta_t)
    return result / 100.0

u_nominal = control.LQR(model, M, L, z_start, t_start, z_final, t_final, delta_t)
z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

lqr_cost = []
for i in range(4):
    print('%d iteration started' % (i+1))
    u = builder.improve(z_nominal, u_nominal)
    z = trajectory(model, fk, z_start, u, delta_t)
    lqr_cost += [full_cost(u, z)]
    z_nominal = z
    u_nominal = u

u_nominal = u_nominal = control.Empty(model, fk, z_start, t_start, t_final, delta_t)
z_nominal = trajectory(model, fk, z_start, u_nominal, delta_t)

empty_cost = []
for i in range(15):
    print('%d iteration started' % (i+1))
    u = builder.improve(z_nominal, u_nominal)
    z = trajectory(model, fk, z_start, u, delta_t)
    empty_cost += [full_cost(u, z)]
    z_nominal = z
    u_nominal = u
print(lqr_cost)
print(empty_cost)
#plt.plot(list(range(1, len(lqr_cost)+1)), [c - lqr_cost[len(lqr_cost) - 1] for c in lqr_cost])
#plt.plot(list(range(1, len(empty_cost)+1)), [c - empty_cost[len(empty_cost) - 1] for c in empty_cost])
#plt.show()
