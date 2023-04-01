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


e_obstacle = sympy.IndexedBase('e_obstacle')
r_obstacle = sympy.Symbol('r_obstacle')

e_obstacle_vec = sympy.Matrix([
    [e_obstacle[i]]
    for i in range(2)
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

print(((e_obstacle_vec.T * end_effector)[0,0] - r_obstacle)**(-2))