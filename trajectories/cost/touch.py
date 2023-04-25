from common import Model, Matrix, Vector
from cost import Cost

import numpy

class Touch(Cost):
    def __init__(self, model: Model, target: Vector):
        self.model = model
        self.target = target

    def cost(self, z:Vector, step:float):
        e_target = self.target
        l = self.model.l
        return numpy.array([[(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))**2]])
    
    def d_cost(self, z:Vector, step:float):
        e_target = self.target
        l = self.model.l
        return numpy.array([[(2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)), (2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)), 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2), 0, 0, 0]])
    
    def dd_cost(self, z:Vector, step:float):
        e_target = self.target
        l = self.model.l
        return numpy.array([[(((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + ((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)) + (numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)**(3/2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.sin(z[0])*l[0] + (-numpy.cos(z[0])*l[0] - numpy.cos(z[1])*l[1] - numpy.cos(z[2])*l[2] + e_target[0])*numpy.cos(z[0])*l[0] + numpy.sin(z[0])**2*l[0]**2 + numpy.cos(z[0])**2*l[0]**2)/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.sin(z[0])*l[0] + (-numpy.cos(z[0])*l[0] - numpy.cos(z[1])*l[1] + e_target[0])*numpy.cos(z[0])*l[0] + numpy.sin(z[0])**2*l[0]**2 + numpy.cos(z[0])**2*l[0]**2)/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)), (2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + ((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)) + (numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)**(3/2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(numpy.sin(z[0])*numpy.sin(z[1])*l[0]*l[1] + numpy.cos(z[0])*numpy.cos(z[1])*l[0]*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + 2*(numpy.sin(z[0])*numpy.sin(z[1])*l[0]*l[1] + numpy.cos(z[0])*numpy.cos(z[1])*l[0]*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)), (2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + (2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(numpy.sin(z[0])*numpy.sin(z[2])*l[0]*l[2] + numpy.cos(z[0])*numpy.cos(z[2])*l[0]*l[2])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)), 0, 0, 0], [(((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + ((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)) + (numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)**(3/2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(numpy.sin(z[0])*numpy.sin(z[1])*l[0]*l[1] + numpy.cos(z[0])*numpy.cos(z[1])*l[0]*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + 2*(numpy.sin(z[0])*numpy.sin(z[1])*l[0]*l[1] + numpy.cos(z[0])*numpy.cos(z[1])*l[0]*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)), (((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + ((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)) + (numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)**(3/2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.sin(z[1])*l[1] + (-numpy.cos(z[0])*l[0] - numpy.cos(z[1])*l[1] - numpy.cos(z[2])*l[2] + e_target[0])*numpy.cos(z[1])*l[1] + numpy.sin(z[1])**2*l[1]**2 + numpy.cos(z[1])**2*l[1]**2)/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.sin(z[1])*l[1] + (-numpy.cos(z[0])*l[0] - numpy.cos(z[1])*l[1] + e_target[0])*numpy.cos(z[1])*l[1] + numpy.sin(z[1])**2*l[1]**2 + numpy.cos(z[1])**2*l[1]**2)/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2)), (2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + (2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(numpy.sin(z[1])*numpy.sin(z[2])*l[1]*l[2] + numpy.cos(z[1])*numpy.cos(z[2])*l[1]*l[2])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)), 0, 0, 0], [2*(((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + ((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[0])*l[0] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[0])*l[0])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(numpy.sin(z[0])*numpy.sin(z[2])*l[0]*l[2] + numpy.cos(z[0])*numpy.cos(z[2])*l[0]*l[2])*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2), 2*(((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + ((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[1])*l[1] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[1])*l[1])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*(numpy.sin(z[1])*numpy.sin(z[2])*l[1]*l[2] + numpy.cos(z[1])*numpy.cos(z[2])*l[1]*l[2])*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2), 2*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2)**(3/2) + 2*((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.cos(z[2])*l[2] - (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])*numpy.sin(z[2])*l[2])**2/((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2) + 2*(numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] - e_target[0])**2) + numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2))*(-(numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])*numpy.sin(z[2])*l[2] + (-numpy.cos(z[0])*l[0] - numpy.cos(z[1])*l[1] - numpy.cos(z[2])*l[2] + e_target[0])*numpy.cos(z[2])*l[2] + numpy.sin(z[2])**2*l[2]**2 + numpy.cos(z[2])**2*l[2]**2)/numpy.sqrt((numpy.sin(z[0])*l[0] + numpy.sin(z[1])*l[1] + numpy.sin(z[2])*l[2] - e_target[1])**2 + (numpy.cos(z[0])*l[0] + numpy.cos(z[1])*l[1] + numpy.cos(z[2])*l[2] - e_target[0])**2), 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])