from common import Vector, Matrix, Model
from common import angles
from common import t_start, z_start, t_final, l, m, I, g, delta_t
import graphic

import numpy as np
import matplotlib.pyplot as plt

l = [0.7, 0.7, 1.6]
model = Model(l, m, I, g)
target = Vector([1.0, 1.0])

exh_l = np.linspace(0, l[2], 10)
exh_a = np.linspace(-np.pi, np.pi, 10)
exh = [
    (ll, aa)
    for aa in np.linspace(-np.pi, np.pi, 10)
    for ll in np.linspace(0, l[2], 10)
]

fig = plt.figure()
ax = fig.add_subplot(111)

for iter in exh:
    ll = iter[0]
    aa = iter[1]

    e2_pos = Vector([target[0]+ll*np.cos(aa), target[1]+ll*np.sin(aa)])
    angl = angles(model, e2_pos)
    if angl is None:
        continue
    if aa > 0:
        angl_2 = aa - np.pi
    else:
        angl_2 = aa + np.pi
    z = Vector([angl[0][0], angl[0][1], angl_2])
    x,y = graphic.pendulum_line(model.l, z)
    ax.plot(x, y, '-o', c='C1')

ax.plot([target[0]], [target[1]], 'o', c='C0')
plt.show()


#print(angles(model, target))