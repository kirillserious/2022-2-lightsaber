from typing import List, Tuple, Union, Generator

import numpy as np
import matplotlib.animation as animation
import matplotlib.figure as figure

from common import Vector

def pendulum_line(
    l:Union[List[float], Vector],
    z:Union[List[float], Vector],
)->Tuple[List[float], List[float]]:
    joint_count = 0
    if isinstance(l, list):
        joint_count = len(l)
    else:
        joint_count, = l.shape
    
    x = [0] + [
        sum([l[k] * np.cos(z[k]) for k in range(j+1)]) 
        for j in range(joint_count)
    ]
    y = [0] + [
        sum([l[k] * np.sin(z[k]) for k in range(j+1)]) 
        for j in range(joint_count)
    ]

    return x, y


def add_pendulum_lines(
    axes,
    l: Union[List[float], Vector],
    z: Union[List[float], Vector],
    lines=15,
) -> None:
    if isinstance(l, list):
        l_len = len(l)
    else:
        l_len, = l.shape
    
    l_len = sum([l[i] for i in range(l_len)])
    axes.set_xlim(-l_len-1, l_len+1)
    axes.set_ylim(-l_len-1, l_len+1)


    axes.set_aspect('equal', adjustable='box')
    axes.grid(True)

    if isinstance(z, list):
        N = len(z)
    else:
        N = z.shape[0]
    
    step = int(N / lines)
    for i in range(lines):
        index = i*step
        x, y = pendulum_line(l, z[index])
        axes.plot(x, y, 'o-', c='C0' ,alpha=1/lines*(i+1))
    x, y = pendulum_line(l, z[N-1])
    axes.plot(x, y, 'o-', c='C0')


def time_range(
    t:Union[List[float], Vector],
    interval: int, #ms
) -> Generator[int, None, None]:
    t_start = t[0]
    if isinstance(t, list):
        t_final = t[len(t) - 1]
    else:
        t_final = t[t.shape[0] - 1]
    
    t_index = 0
    for t_current in np.arange(t_start, t_final, 0.001 * interval):
        if t[t_index] >= t_current:
            yield t_index
        else:
            t_index += 1
            yield t_index        
        

def PendulumAnimation(
    figure: figure.Figure,
    axes,
    l, # vector of lengths
    z, # z[i] is a vector of i-th position
    t, # t[i] is i-th time moment
)->animation.FuncAnimation:
    if isinstance(l, list):
        l_len = len(l)
    else:
        l_len, = l.shape
    
    l_len = sum([l[i] for i in range(l_len)])
    axes.set_xlim(-l_len-1, l_len+1)
    axes.set_ylim(-l_len-1, l_len+1)


    axes.set_aspect('equal', adjustable='box')
    axes.grid(True)

    line, = axes.plot([], [], 'o-', lw=2)
    text = axes.text(0.05, 0.9, '', transform=axes.transAxes)

    def init():
        return [line, text]

    def animate(time_index):
        x,y = pendulum_line(l, z[time_index])
        line.set_data(x, y)
        text.set_text('$t$ = %.2fs' % (t[time_index]))
        return [line, text]

    return animation.FuncAnimation(
        figure,
        animate,
        time_range(t, 1),
        interval=1,
        blit=True,
        init_func=init,
    )


