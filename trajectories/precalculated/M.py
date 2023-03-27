import numpy as np

from common import Model, Vector, Matrix

def M(model: Model, q: Vector)->Matrix:
    l = model.l
    m = model.m
    I = model.I
    g = model.g

    return Matrix([
    [
        I[0] + l[0]**2*(m[0]/4 + m[1] + m[2]),
        l[0]*l[1]*(m[1]/2 - m[2])*np.cos(q[0] - q[1]),
        l[0]*l[2]*m[2]*np.cos(q[0] - q[2])/2
    ],
    [
        l[0]*l[1]*(m[1]/2 - m[2])*np.cos(q[0] - q[1]),
        I[1] + l[1]**2*m[1]/4 + l[1]**2*m[2],
        l[1]*l[2]*m[2]*np.cos(q[1] - q[2])/2
    ],
    [
        l[0]*l[2]*m[2]*np.cos(q[0] - q[2])/2,
        l[1]*l[2]*m[2]*np.cos(q[1] - q[2])/2,
        I[2] + l[2]**2*m[2]/4
    ]
])