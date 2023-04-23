from common import Vector, Matrix
from .cost import Cost

class TargetFinal(Cost):
    def __init__(self, z_final: Vector):
        self.z_final = z_final

    def cost(self, z: Vector, step)->float:
        z_final = self.z_final
        #return (z-z_final).T.dot(z-z_final)
        return (z[0] - z_final[0])**2 + (z[1] - z_final[1])**2 + (z[2] - z_final[2])**2 + (z[3] - z_final[3])**2 + (z[4] - z_final[4])**2 + (z[5] - z_final[5])**2
    
    def d_cost(self, z: Vector, step)->Vector:
        z_final = self.z_final
        return Vector([2*z[0] - 2*z_final[0], 2*z[1] - 2*z_final[1], 2*z[2] - 2*z_final[2], 2*z[3] - 2*z_final[3], 2*z[4] - 2*z_final[4], 2*z[5] - 2*z_final[5]])
    
    def dd_cost(self, z: Vector, step)->Matrix:
        return Matrix([[2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 2]])