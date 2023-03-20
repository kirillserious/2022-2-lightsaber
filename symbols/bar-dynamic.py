import sympy

m = sympy.symbols('m')
g = sympy.symbols('g')
l = sympy.symbols('l')
I = sympy.symbols('I')

t = sympy.symbols('t')
q = [sympy.Function('q'+str(i))(t) for i in range(0, 3)]
dot_q = [
    sympy.diff(q[i], t)
    for i in range(0, 3)
]

v_sqr = sympy.diff(l * sympy.cos(q[2])/2 + q[0], t) ** 2 + sympy.diff(l * sympy.sin(q[2])/2 + q[1], t) ** 2
omega_sqr = dot_q[2] ** 2
K = m * v_sqr / 2 + I * omega_sqr / 2
Pi = m * g * (sympy.sin(q[2]) * l /2 + q[1])
L = K - Pi

tau = [
    sympy.diff(sympy.diff(L, dot_q[i]), t) - sympy.diff(L, q[i])
    for i in range(0, 3)
]

print(sympy.simplify(tau[0]))
