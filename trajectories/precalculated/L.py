import numpy as np

from common import Vector, Model

def L(model: Model, z: Vector)->Vector:
    l = model.l
    m = model.m
    I = model.I
    g = model.g
    return Vector([
              g*l[0]*m[0]*np.cos(z[0])/2
            + g*l[0]*m[1]*np.cos(z[0])
            + g*l[0]*m[2]*np.cos(z[0])
            + l[0]*l[1]*m[1]*np.sin(z[0] - z[1])*z[4]**2/2
            + l[0]*l[1]*m[2]*np.sin(z[0] - z[1])*z[4]**2
            + l[0]*l[2]*m[2]*np.sin(z[0] - z[2])*z[5]**2/2
        ,
              g*l[1]*m[1]*np.cos(z[1])/2
            + g*l[1]*m[2]*np.cos(z[1])
            - l[0]*l[1]*m[1]*np.sin(z[0] - z[1])*z[3]**2/2
            - l[0]*l[1]*m[2]*np.sin(z[0] - z[1])*z[3]**2
            + l[1]*l[2]*m[2]*np.sin(z[1] - z[2])*z[5]**2/2
        ,
              g*l[2]*m[2]*np.cos(z[2])/2
            - l[0]*l[2]*m[2]*np.sin(z[0] - z[2])*z[3]**2/2
            - l[1]*l[2]*m[2]*np.sin(z[1] - z[2])*z[4]**2/2
    ])