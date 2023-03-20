from common import Vector

# Входные параметры

# Параметры модели
g = 9.8
m = [0.1, 0.1, 0.1]
l = [1, 1, 1]
I = [m[i] * l[i] * l[i] / 3 for i in range(3)]

# Параметры задачи управления
t_start = 0.0
t_final = 1.0
z_start = Vector([1.4, 1.4, 1.4, 0, 0, 0])

# Параметры дискретизации
delta_t = 0.001



