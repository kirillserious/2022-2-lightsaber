#!/usr/bin/python3

data = '''
(l0*l1*((1/2)*m1 - m2)*(-Math.pow(l0, 2)*l1*l2*m2*((1/2)*m1 - m2)*Math.cos(z0 - z1)*Math.cos(z0 - z2)/(2*I0 + 2*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/2)*l1*l2*m2*Math.cos(z1 - z2))*Math.cos(z0 - z1)/((I0 + Math.pow(l0, 2)*((1/4)*m0 + m1 + m2))*(I1 - Math.pow(l0, 2)*Math.pow(l1, 2)*Math.pow((1/2)*m1 - m2, 2)*Math.pow(Math.cos(z0 - z1), 2)/(I0 + Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l1, 2)*m1 + Math.pow(l1, 2)*m2)) - l0*l2*m2*Math.cos(z0 - z2)/(2*I0 + 2*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)))*(-1/2*g*l0*m0*Math.cos(z0) - g*l0*m1*Math.cos(z0) - g*l0*m2*Math.cos(z0) - 1/2*l0*l1*m1*Math.pow(z4, 2)*Math.sin(z0 - z1) - l0*l1*m2*Math.pow(z4, 2)*Math.sin(z0 - z1) - 1/2*l0*l2*m2*Math.pow(z5, 2)*Math.sin(z0 - z2) + u0)/(I2 - Math.pow(l0, 2)*Math.pow(l2, 2)*Math.pow(m2, 2)*Math.pow(Math.cos(z0 - z2), 2)/(4*I0 + 4*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l2, 2)*m2 - Math.pow(-Math.pow(l0, 2)*l1*l2*m2*((1/2)*m1 - m2)*Math.cos(z0 - z1)*Math.cos(z0 - z2)/(2*I0 + 2*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/2)*l1*l2*m2*Math.cos(z1 - z2), 2)/(I1 - Math.pow(l0, 2)*Math.pow(l1, 2)*Math.pow((1/2)*m1 - m2, 2)*Math.pow(Math.cos(z0 - z1), 2)/(I0 + Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l1, 2)*m1 + Math.pow(l1, 2)*m2)) + (Math.pow(l0, 2)*l1*l2*m2*((1/2)*m1 - m2)*Math.cos(z0 - z1)*Math.cos(z0 - z2)/(2*I0 + 2*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) - 1/2*l1*l2*m2*Math.cos(z1 - z2))*(-1/2*g*l1*m1*Math.cos(z1) - g*l1*m2*Math.cos(z1) + (1/2)*l0*l1*m1*Math.pow(z3, 2)*Math.sin(z0 - z1) + l0*l1*m2*Math.pow(z3, 2)*Math.sin(z0 - z1) - 1/2*l1*l2*m2*Math.pow(z5, 2)*Math.sin(z1 - z2) + u1)/((I1 - Math.pow(l0, 2)*Math.pow(l1, 2)*Math.pow((1/2)*m1 - m2, 2)*Math.pow(Math.cos(z0 - z1), 2)/(I0 + Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l1, 2)*m1 + Math.pow(l1, 2)*m2)*(I2 - Math.pow(l0, 2)*Math.pow(l2, 2)*Math.pow(m2, 2)*Math.pow(Math.cos(z0 - z2), 2)/(4*I0 + 4*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l2, 2)*m2 - Math.pow(-Math.pow(l0, 2)*l1*l2*m2*((1/2)*m1 - m2)*Math.cos(z0 - z1)*Math.cos(z0 - z2)/(2*I0 + 2*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/2)*l1*l2*m2*Math.cos(z1 - z2), 2)/(I1 - Math.pow(l0, 2)*Math.pow(l1, 2)*Math.pow((1/2)*m1 - m2, 2)*Math.pow(Math.cos(z0 - z1), 2)/(I0 + Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l1, 2)*m1 + Math.pow(l1, 2)*m2))) + (-1/2*g*l2*m2*Math.cos(z2) + (1/2)*l0*l2*m2*Math.pow(z3, 2)*Math.sin(z0 - z2) + (1/2)*l1*l2*m2*Math.pow(z4, 2)*Math.sin(z1 - z2) + u2)/(I2 - Math.pow(l0, 2)*Math.pow(l2, 2)*Math.pow(m2, 2)*Math.pow(Math.cos(z0 - z2), 2)/(4*I0 + 4*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l2, 2)*m2 - Math.pow(-Math.pow(l0, 2)*l1*l2*m2*((1/2)*m1 - m2)*Math.cos(z0 - z1)*Math.cos(z0 - z2)/(2*I0 + 2*Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/2)*l1*l2*m2*Math.cos(z1 - z2), 2)/(I1 - Math.pow(l0, 2)*Math.pow(l1, 2)*Math.pow((1/2)*m1 - m2, 2)*Math.pow(Math.cos(z0 - z1), 2)/(I0 + Math.pow(l0, 2)*((1/4)*m0 + m1 + m2)) + (1/4)*Math.pow(l1, 2)*m1 + Math.pow(l1, 2)*m2))
'''

for symbol in ['m', 'l', 'I', 'q', 'z', 'u']:
    for number in ['0', '1', '2', '3', '4', '5']:
        data = data.replace(f'{symbol}{number}', f'{symbol}[{number}]')
data = data.replace('cos', 'np.cos')
data = data.replace('sin', 'np.sin')
print(data)