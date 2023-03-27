#!/usr/bin/python3

import sympy

n, m = sympy.symbols('n m')
A_k = sympy.MatrixSymbol('A_k', n, n)
B_k = sympy.MatrixSymbol('B_k', n, m)

x_k = sympy.MatrixSymbol('x_k', n, 1)
u_k = sympy.MatrixSymbol('u_k', m, 1)
delta_x_k = sympy.MatrixSymbol('\delta x_k', n, 1)
delta_u_k = sympy.MatrixSymbol('\delta u_k', m, 1)

print(A_k * delta_x_k + B_k * delta_u_k)