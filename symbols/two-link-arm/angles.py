import sympy

q = sympy.IndexedBase('q')
l = sympy.IndexedBase('l')
x = sympy.symbols('x')
y = sympy.symbols('y')

#x = sympy.cos(q[0])*l[0] + sympy.cos(q[1])*l[1]
#y = sympy.sin(q[0])*l[0] + sympy.sin(q[1])*l[1]
res = sympy.solve([
    sympy.cos(q[0])*l[0] + sympy.cos(q[1])*l[1] - x,
    sympy.sin(q[0])*l[0] + sympy.sin(q[1])*l[1] - y,
], [
    q[0], q[1]
], dict=True)
print(res)