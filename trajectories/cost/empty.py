from common import Vector, Matrix
from cost import Cost

import numpy as np

class Empty(Cost):
    def __init__(self):
        pass

    def cost(self, z:Vector, step:float):
        return 0.0
    
    def d_cost(self, z:Vector, step:float):
        return np.zeros((z.shape[0]))
    
    def dd_cost(self, z:Vector, step:float):
        return np.zeros((z.shape[0], z.shape[0]))