#!/usr/bin/python3

import sympy

# Параметры программы
links_count = 2
calculate_inertia = False #True

# Константы
g = sympy.Symbol('g') # Ускорение свободного падения

# Параметры задачи
## Длина каждого звена
l = sympy.IndexedBase('l')
## Масса каждого звена 
m = sympy.IndexedBase('m')

## Моменты инерции каждого звена
if calculate_inertia:
    I = [
        m[i] * l[i] * l[i] / 3
        for i in range(0, links_count)
    ]
else:
    I = sympy.IndexedBase('I')

t = sympy.symbols('t')
q = sympy.IndexedBase('q')

M = sympy.matrices.Matrix([
    [I[0] + l[0]**2*m[0]/4 + l[0]**2*m[1], l[0]*l[1]*m[1]*sympy.cos(q[0] - q[1])/2],
    [l[0]*l[1]*m[1]*sympy.cos(q[0] - q[1])/2, I[1] + l[1]**2*m[1]/4]
])

L = sympy.matrices.Matrix([
    g*sympy.cos(q[0])*l[0]*m[0]/2 + g*sympy.cos(q[0])*l[0]*m[1] + sympy.sin(q[0] - q[1])*Derivative(q[1](t), t)**2*l[0]*l[1]*m[1]/2
])