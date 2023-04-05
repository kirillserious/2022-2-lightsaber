import sympy

# Параметры программы
links_count = 3
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
q_vec = sympy.Matrix([
    [q[i]] for i in range(links_count*2)
])
obstacle = sympy.IndexedBase('obstacle')
r_obstacle =sympy.Symbol('r_obstacle')

end_effector = [
    sum([
        l[i] * sympy.cos(q[i])
        for i in range(links_count)
    ]),
    sum([
        l[i] * sympy.sin(q[i])
        for i in range(links_count)
    ]),
]
dist_sqr = (end_effector[0] - obstacle[0])**2 + (end_effector[1] - obstacle[1])**2
from sympy.printing.numpy import NumPyPrinter 

cost = (dist_sqr - r_obstacle)**(-2)
d_cost = sympy.Matrix([[cost]]).jacobian(q_vec)
dd_cost = d_cost.jacobian(q_vec)
code = NumPyPrinter().doprint(dd_cost)
print(code)
exit()
d_cost = sympy.Matrix([[cost]]).jacobian(q_vec)
dd_cost = d_cost.jacobian(q_vec)
print(dd_cost)