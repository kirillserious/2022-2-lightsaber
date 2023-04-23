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

z_final = sympy.IndexedBase('z_final')

cost = sum([
    (q[i] - z_final[i])**2  
    for i in range(links_count* 2)
])
#print(cost)

cost = sympy.Matrix([[cost]])
q_vec = [
    q[i]
    for i in range(links_count*2)
]
d_cost = cost.jacobian(q_vec)
#print(d_cost)

dd_cost = d_cost.jacobian(q_vec)
print(dd_cost)