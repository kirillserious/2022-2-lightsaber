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

# Фазовые переменные
t = sympy.symbols('t')
q = [sympy.Function('q['+str(i)+']')(t) for i in range(0, links_count)]
dot_q = [
    sympy.diff(q[i], t)
    for i in range(0, links_count)
]

# Положение центра каждого звена
x_center = [
    sum([l[j] * sympy.cos(q[j]) for j in range(0, i)]) + l[i]/2*sympy.cos(q[i])
    for i in range(0, links_count)
]

y_center = [
    sum([l[j]*sympy.sin(q[j]) for j in range(0, i)]) + l[i]/2*sympy.sin(q[i])
    for i in range(0, links_count)
]

# Скорость центра каждого звена в квадрате
v_square = [
    sympy.diff(x_center[i], t)**2 + sympy.diff(y_center[i], t)**2
    for i in range(0, links_count)
]

# Скорость вращения каждого звена
omega = [
    dot_q[i]
    for i in range(0, links_count)
]

# Общая кинетическая энегия системы
K = sum([
    m[i] * v_square[i]/2 + I[i] * omega[i] * omega[i] / 2 
    for i in range(0, links_count)
])

# Общая потенциальная энергия системы
Pi = sum([
    m[i] * g * y_center[i]
    for i in range(0, links_count)
])

# Лангражиан
L = K - Pi

# Моменты внешних сил
tau = [
    sympy.diff(sympy.diff(L, dot_q[i]), t) - sympy.diff(L, q[i])
    for i in range(0, links_count)
]

#print(sympy.simplify(sympy.collect(tau[0], sympy.diff(q[0], (t,2))).coeff(sympy.diff(q[0], (t,2)), 1)))
#print()

print('Уравнение динамики:\n')
for i in range(0, links_count):
    print('tau%d = '%(i) + str(sympy.simplify(tau[i])))
    print()
print()

z = sympy.IndexedBase('z')
dot_z = sympy.IndexedBase('dz')

for i in range(links_count):
    for j in range(links_count):
        tau[i] = tau[i].subs(sympy.diff(q[j], (t,2)), dot_z[links_count + j])
        tau[i] = tau[i].subs(sympy.diff(q[j], t), z[links_count + j])
        tau[i] = tau[i].subs(q[j], z[j])

print('Уравнение динамики в терминах z:\n')
for i in range(0, links_count):
    print('tau%d = '%(i) + str(sympy.simplify(tau[i])))
    print()
print()

print(sympy.simplify(tau[0].collect(dot_z[2])))