#!/usr/bin/python3

import sympy

links_count = 3

z = sympy.symbols('z(:%d)'%(links_count*2))
u = sympy.symbols('u(:%d)'%(links_count))

z_final = sympy.symbols('z_final(:%d)'%(links_count*2))


z_vec = sympy.Matrix([[z[i]] for i in range(links_count*2)])
u_vec = sympy.Matrix([[u[i]] for i in range(links_count)])
z_final_vec = sympy.Matrix([[z_final[i]] for i in range(links_count*2)])

l_final = (z_vec - z_final_vec).transpose() * (z_vec - z_final_vec)
l = u_vec.transpose() * u_vec


l_final_z = l_final.jacobian(z)
l_final_zz = l_final.jacobian(z).jacobian(z)
print(l_final_z)
print(l_final_zz)
print()

l_zz = l.jacobian(z).jacobian(z)
l_uu = l.jacobian(u).jacobian(u)
l_uz = l.jacobian(u).jacobian(z)

print(l_zz)
print(l_uu)
print(l_uz)