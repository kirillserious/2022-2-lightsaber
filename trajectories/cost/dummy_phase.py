from common import Vector, Matrix
from cost import Cost

import numpy as np

class DummyPhase(Cost):
    def __init__(self):
        pass

    def cost(self, z:Vector, step:float):
        return z.T.dot(z)
    
    def d_cost(self, z:Vector, step:float):
        return z
    
    def dd_cost(self, z:Vector, step:float):
        return np.eye(z.shape[0], z.shape[0])