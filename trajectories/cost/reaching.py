from common import Model, Matrix, Vector
from cost import Cost

import numpy as np

class Reaching(Cost):
    def __init__(self, model: Model, target: Vector):
        self.model = model
        self.target = target

    def cost(self, z: Vector, step: float)->float:
        l = self.model.l
        e_target = self.target
        return (np.sin(z[0])*l[0] + np.sin(z[1])*l[1] + np.sin(z[2])*l[2] - e_target[1])**2 + (np.cos(z[0])*l[0] + np.cos(z[1])*l[1] + np.cos(z[2])*l[2] - e_target[0])**2
    
    def d_cost(self, z: Vector, step: float)->Vector:
        l = self.model.l
        e_target = self.target
        return Vector([2*(np.sin(z[0])*l[0] + np.sin(z[1])*l[1] + np.sin(z[2])*l[2] - e_target[1])*np.cos(z[0])*l[0] - 2*(np.cos(z[0])*l[0] + np.cos(z[1])*l[1] + np.cos(z[2])*l[2] - e_target[0])*np.sin(z[0])*l[0], 2*(np.sin(z[0])*l[0] + np.sin(z[1])*l[1] + np.sin(z[2])*l[2] - e_target[1])*np.cos(z[1])*l[1] - 2*(np.cos(z[0])*l[0] + np.cos(z[1])*l[1] + np.cos(z[2])*l[2] - e_target[0])*np.sin(z[1])*l[1], 2*(np.sin(z[0])*l[0] + np.sin(z[1])*l[1] + np.sin(z[2])*l[2] - e_target[1])*np.cos(z[2])*l[2] - 2*(np.cos(z[0])*l[0] + np.cos(z[1])*l[1] + np.cos(z[2])*l[2] - e_target[0])*np.sin(z[2])*l[2], 0, 0, 0])

    def dd_cost(self, z: Vector, step: float)->Matrix:
        l = self.model.l
        e_target = self.target
        return Matrix([[-(2*np.sin(z[0])*l[0] + 2*np.sin(z[1])*l[1] + 2*np.sin(z[2])*l[2] - 2*e_target[1])*np.sin(z[0])*l[0] + (-2*np.cos(z[0])*l[0] - 2*np.cos(z[1])*l[1] - 2*np.cos(z[2])*l[2] + 2*e_target[0])*np.cos(z[0])*l[0] + 2*np.sin(z[0])**2*l[0]**2 + 2*np.cos(z[0])**2*l[0]**2, 2*np.sin(z[0])*np.sin(z[1])*l[0]*l[1] + 2*np.cos(z[0])*np.cos(z[1])*l[0]*l[1], 2*np.sin(z[0])*np.sin(z[2])*l[0]*l[2] + 2*np.cos(z[0])*np.cos(z[2])*l[0]*l[2], 0, 0, 0], [2*np.sin(z[0])*np.sin(z[1])*l[0]*l[1] + 2*np.cos(z[0])*np.cos(z[1])*l[0]*l[1], -(2*np.sin(z[0])*l[0] + 2*np.sin(z[1])*l[1] + 2*np.sin(z[2])*l[2] - 2*e_target[1])*np.sin(z[1])*l[1] + (-2*np.cos(z[0])*l[0] - 2*np.cos(z[1])*l[1] - 2*np.cos(z[2])*l[2] + 2*e_target[0])*np.cos(z[1])*l[1] + 2*np.sin(z[1])**2*l[1]**2 + 2*np.cos(z[1])**2*l[1]**2, 2*np.sin(z[1])*np.sin(z[2])*l[1]*l[2] + 2*np.cos(z[1])*np.cos(z[2])*l[1]*l[2], 0, 0, 0], [2*np.sin(z[0])*np.sin(z[2])*l[0]*l[2] + 2*np.cos(z[0])*np.cos(z[2])*l[0]*l[2], 2*np.sin(z[1])*np.sin(z[2])*l[1]*l[2] + 2*np.cos(z[1])*np.cos(z[2])*l[1]*l[2], -(2*np.sin(z[0])*l[0] + 2*np.sin(z[1])*l[1] + 2*np.sin(z[2])*l[2] - 2*e_target[1])*np.sin(z[2])*l[2] + (-2*np.cos(z[0])*l[0] - 2*np.cos(z[1])*l[1] - 2*np.cos(z[2])*l[2] + 2*e_target[0])*np.cos(z[2])*l[2] + 2*np.sin(z[2])**2*l[2]**2 + 2*np.cos(z[2])**2*l[2]**2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
