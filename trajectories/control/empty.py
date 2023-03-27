from typing import Tuple, Callable, Optional, List

from tqdm import tqdm

from common import Model, Vector, Matrix


# fcn(model, zk, uk, step)
VectorFcn = Callable[[Model, Vector, Vector, float], Vector]

def Empty(
    model: Model,
    fk: VectorFcn,
    z_start: Vector,
    t_start: float,
    t_final: float,
    step: Optional[float]=0.001,
)->Tuple[List[Vector], List[Vector]]:

    N = int((t_final - t_start) / step)

    print('Initialize empty control')
    u = [Vector([0.0, 0.0, 0.0]) for i in tqdm(range(N))]
    for i in range(int(N/6*4), int(N/6*5)):
        u[i] = Vector([10.0, -10.0, -10.0])
    for i in range(int(N/6*5), N):
        u[i] = Vector([-10.0, 10.0, 10.0])

    print('Initialize empty trajectory')
    z = [None] * N
    z[0] = z_start
    for i in tqdm(range(N-1)):
        z[i+1] = fk(model, z[i], u[i], step)
    
    return z, u