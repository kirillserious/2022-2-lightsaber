from typing import List, Tuple, Union, Generator

import numpy as np
import matplotlib.figure as figure

from common import Vector, Matrix

def state_lines(
    axes,
    zs: Union[List[Matrix], List[List[Vector]]],
):
    for z in zs:
        if isinstance(z, list):
            N = len(z)
        else:
            N = z.shape[0]
        theta = [
            [ z[j][i] for j in range(N)]
            for i in range(3)
        ]
        axes.plot3D(theta[0], theta[1], theta[2])

def end_effector_lines(
    axes,
    l: Union[List[float], Vector],
    zs: Union[List[Matrix], List[List[Vector]]],
    t_from: float,
    step: float,
    color:str='C0',
):
    for i in range(len(zs)):
        z = zs[i]
        if isinstance(z, list):
            N = len(z)
        else:
            N = z.shape[0]
        theta = [
            [ z[j][i] for j in range(N)]
            for i in range(3)
        ]
        end_effector = [
            [
                sum([l[k]*np.cos(theta[k][i]) for k in range(len(theta))])
                for i in range(N)
            ],
            [
                sum([l[k]*np.sin(theta[k][i]) for k in range(len(theta))])
                for i in range(N)
            ],
        ]
        
        t = [
            t_from + i*step
            for i in range(len(theta[0]))
        ]

        axes.plot3D(t, end_effector[0], end_effector[1], c=color, alpha=1/len(zs)*(i+1))


