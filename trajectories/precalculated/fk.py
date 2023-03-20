from common import Vector, Matrix
from common import delta_t, g, m, l, I

import numpy as np

from precalculated import f

def fk(zk: Vector, uk: Vector)->Vector:
    '''
        fk - функция в дискретном ОДУ
        z_{k+1} = f(z^k, u^k)
    '''
    return delta_t * f(zk, uk) + zk