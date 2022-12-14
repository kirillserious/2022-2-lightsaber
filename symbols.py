import sympy

# Параметры программы
links_count = 3
calculate_inertia = True

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
q = [sympy.Function('q'+str(i))(t) for i in range(0, links_count)]
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
    sympy.diff(x_center[i], t) + sympy.diff(y_center[i], t)
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

for i in range(0, links_count):
    print('tau%d = '%(i) + str(tau[i]))

