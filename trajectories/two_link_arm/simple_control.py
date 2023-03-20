#!/usr/bin/python3

import numpy as np
from tqdm import tqdm
Vector = np.array
Matrix = np.array

# Входные параметры

# Параметры модели
g = 9.8
m = [0.1, 0.1]
l = [1, 1]
I = [m[i] * l[i] * l[i] / 3 for i in range(2)]

# Параметры задачи управления
t_start = 0.0
t_final = 1.0
z_start = Vector([0.0, 0.0, 0.0, 0.0])
z_final = Vector([1.366025, -0.366025, 0.0, 0.0])

# Параметры дискретизации
delta_t = 0.001
N = int(np.ceil((t_final - t_start) / delta_t))

t = np.zeros(N+1)
t[0] = t_start
for i in range(0, N):
    t[i+1] = t[i] + delta_t

A_mat = delta_t * Matrix([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
]) + np.eye(4)

B_mat = delta_t * Matrix([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

T_mat = 100000.0 * Matrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

M_mat =  delta_t * Matrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

N_mat =  delta_t * Matrix([
    [1, 0],
    [0, 1]
])

print('Backward calculations')
P_mat_list = np.zeros((N+1, 4, 4))
P_mat_list[N] = T_mat
for i in tqdm(range(N, 0, -1)):
    A_mat_T = np.transpose(A_mat)
    B_mat_T = np.transpose(B_mat)
    first = A_mat_T.dot(P_mat_list[i]).dot(A_mat)
    second = N_mat + B_mat_T.dot(P_mat_list[i]).dot(B_mat)
    second = np.linalg.inv(second)
    second = A_mat_T.dot(P_mat_list[i]).dot(B_mat).dot(second).dot(B_mat_T).dot(P_mat_list[i]).dot(A_mat)
    P_mat_list[i-1] = M_mat + first - second

print('Control and trajectory calculations')
v = np.zeros((N, 2))
z = np.zeros((N+1, 4))
z[0] = z_start - z_final
for i in tqdm(range(N)):
    B_mat_T = np.transpose(B_mat)
    tmp = N_mat + B_mat_T.dot(P_mat_list[i]).dot(B_mat)
    tmp = np.linalg.inv(tmp)
    v[i] = - tmp.dot(B_mat_T).dot(P_mat_list[i]).dot(A_mat).dot(z[i])
    z[i+1] = A_mat.dot(z[i]) + B_mat.dot(v[i])
    

print('Translate trajectory')
for i in tqdm(range(N+1)):
    z[i] = z[i] + z_final



# Картинка
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-3, 3))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)


time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    x = [0] + [
        sum([l[k] * np.cos(z[i][k]) for k in range(j+1)]) 
        for j in range(2)
    ]
    y = [0] + [
        sum([l[k] * np.sin(z[i][k]) for k in range(j+1)]) 
        for j in range(2)
    ]  

    line.set_data(x, y)
    time_text.set_text('t = %.2fs' % (t[i]))
    #time_text.set_text(time_template % (res.t[i]))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
                              interval=1, blit=True, init_func=init)


plt.show()