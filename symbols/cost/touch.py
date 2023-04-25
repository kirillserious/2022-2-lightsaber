import sympy
from sympy.printing.numpy import NumPyPrinter

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
q = sympy.IndexedBase('z')

e_target = sympy.IndexedBase('e_target')
e_target_vec = sympy.Matrix([
    [e_target[i]]
    for i in range(2)
])

second_link_effector = sympy.Matrix([
    [
        sum([
            l[i] * sympy.cos(q[i])
            for i in range(links_count-1)
        ])
    ],
    [
        sum([
            l[i] * sympy.sin(q[i])
            for i in range(links_count-1)
        ])
    ],
])

end_effector = sympy.Matrix([
    [
        sum([
            l[i] * sympy.cos(q[i])
            for i in range(links_count)
        ])
    ],
    [
        sum([
            l[i] * sympy.sin(q[i])
            for i in range(links_count)
        ])
    ],
])

dist_1 = sympy.sqrt((end_effector - e_target_vec).T * (end_effector - e_target_vec))
dist_2 = sympy.sqrt((second_link_effector - e_target_vec).T * (second_link_effector - e_target_vec))

cost = (dist_1 + dist_2)**2
printer = NumPyPrinter()
#print(printer.doprint(cost))
q_vec = [q[i] for i in range(links_count*2)]
d_cost = cost.jacobian(q_vec)
#print(printer.doprint(d_cost))
dd_cost = d_cost.jacobian(q_vec)
print(printer.doprint(dd_cost))

