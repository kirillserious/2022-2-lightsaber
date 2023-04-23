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
q = sympy.IndexedBase('z')

e_target = sympy.IndexedBase('e_target')
e_target_vec = sympy.Matrix([
    [e_target[i]]
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

cost = (end_effector - e_target_vec).T * (end_effector - e_target_vec)
#print(cost)
#cost = sympy.Matrix([[cost]])
q_vec = [q[i] for i in range(links_count*2)]
d_cost = cost.jacobian(q_vec)
#print(d_cost)
dd_cost = d_cost.jacobian(q_vec)
#print(dd_cost)

speed = sympy.Matrix([
    [
        -l[j]*sympy.sin(q[j])
        for j in range(links_count)
    ],
    [
        l[j]*sympy.cos(q[j])
        for j in range(links_count)
    ],
]) * sympy.Matrix([
    [q[i+links_count]]
    for i in range(links_count)
])

cost = (speed - e_target_vec).T * (speed - e_target_vec)
#print(cost)
d_cost = cost.jacobian(q_vec)
#print(d_cost)
dd_cost = d_cost.jacobian(q_vec)
print(dd_cost)