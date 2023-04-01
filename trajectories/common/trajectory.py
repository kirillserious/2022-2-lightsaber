from typing import List, Union, Callable

import numpy as np
from tqdm import tqdm

from common import Matrix, Vector, Model

def trajectory(
    model: Model,
    f: Callable[[Model, Vector, Vector, float], Vector],
    z_start: Vector,
    u: Union[Matrix, List[Vector]],
    step: float = 0.001,
) -> Matrix:
    if isinstance(u, list):
        N = len(u) + 1
    else:
        N = u.shape[0] + 1
    
    print('Trajectory calculation')
    z = np.zeros((N, z_start.shape[0]))
    z[0] = z_start
    for k in tqdm(range(N-1)):
        z[k+1] = f(model, z[k], u[k], step)
    return z
