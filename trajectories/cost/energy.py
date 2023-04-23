from common import Vector, Matrix
from cost import Cost

import numpy as np

class Energy(Cost):
    def __init__(self):
        pass

    def cost(self, u: Vector, step: float)->float:
        return u.T.dot(u)
    
    def d_cost(self, u: Vector, step: float)->Vector:
        return u
    
    def dd_cost(self, u: Vector, step: float)->Matrix:
        return np.eye(N=u.shape[0], M=u.shape[0])