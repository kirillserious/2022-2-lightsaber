#!/usr/bin/python3

import sympy

# Параметры программы
links_count = 3
calculate_inertia = False #True

# Константы
g = sympy.Symbol('g') # Ускорение свободного падения

# Параметры задачи
## Длина каждого звена
l = sympy.symbols('l(:%d)'%(links_count))
## Масса каждого звена 
m = sympy.symbols('m(:%d)'%(links_count))
## Моменты инерции каждого звена
if calculate_inertia:
    I = [
        m[i] * l[i] * l[i] / 12
        for i in range(0, links_count)
    ]
else:
    I = [
        sympy.symbols('I' + str(i))
        for i in range(0, links_count)
    ]

# Фазовые переменные
t = sympy.symbols('t')
q = sympy.symbols('z(:%d)'%(links_count*2))

M = sympy.matrices.Matrix([
    [
        I[0] + l[0]**2*(m[0]/4 + m[1] + m[2]),
        l[0]*l[1]*(m[1]/2 - m[2])*sympy.cos(q[0] - q[1]),
        l[0]*l[2]*m[2]*sympy.cos(q[0] - q[2])/2
    ],
    [
        l[0]*l[1]*(m[1]/2 - m[2])*sympy.cos(q[0] - q[1]),
        I[1] + l[1]**2*m[1]/4 + l[1]**2*m[2],
        l[1]*l[2]*m[2]*sympy.cos(q[1] - q[2])/2
    ],
    [
        l[0]*l[2]*m[2]*sympy.cos(q[0] - q[2])/2,
        l[1]*l[2]*m[2]*sympy.cos(q[1] - q[2])/2,
        I[2] + l[2]**2*m[2]/4
    ]
])

print(M.inv(method='LU'))
