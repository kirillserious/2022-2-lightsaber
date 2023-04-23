from typing import Callable
from common import Model, Vector, Matrix

# fcn(zk or uk, step)
ScalarFcn = Callable[[Vector, float], float]
VectorFcn = Callable[[Vector, float], Vector]
MatrixFcn = Callable[[Vector, float], Matrix]

class Cost:
    def __init__(self, cost: ScalarFcn, d_cost: VectorFcn, dd_cost: MatrixFcn):
        self.__cost = cost
        self.__d_cost = d_cost
        self.__dd_cost = dd_cost

    def cost(self, vector: Vector, step: float)->float:
        return self.__cost(vector, step)
    
    def d_cost(self, vector: Vector, step: float)->Vector:
        return self.__d_cost(vector, step)

    def dd_cost(self, vector: Vector, step: float)->Matrix:
        return self.__dd_cost(vector, step)

    def __add__(self, other):
        return Cost(
            cost= lambda vector, step: self.cost(vector, step) + other.cost(vector, step),
            d_cost = lambda vector, step: self.d_cost(vector, step) + other.d_cost(vector, step),
            dd_cost= lambda vector, step: self.dd_cost(vector, step) + other.dd_cost(vector, step),
        )
    
    def __rmul__(self, mul: float):
        return Cost(
            cost = lambda vector, step: mul * self.cost(vector, step),
            d_cost = lambda vector, step: mul * self.d_cost(vector, step),
            dd_cost = lambda vector, step: mul * self.dd_cost(vector, step),
        )