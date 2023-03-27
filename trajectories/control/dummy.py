from typing import Optional, Tuple, Callable

import numpy as np
from tqdm import tqdm

from common import Model, Vector, Matrix

MatrixFcn = Callable[[Model, Vector], Matrix]
VectorFcn = Callable[[Model, Vector], Vector]

def Dummy(
    model:Model,
    M: MatrixFcn,
    L: VectorFcn,
    z_start: Vector,
    t_start: float,
    z_final: Vector,
    t_final: float,
    step: Optional[float] = 0.001,
)->Tuple[Matrix, Matrix]:
    n = len(model.l)
    A_mat = step * np.concatenate(
        (np.concatenate((np.zeros((n,n)), np.eye(n,n)), axis=1), np.zeros((n, 2*n))),
        axis=0,
    )
    A_mat += np.eye(2*n)
    B_mat = step * np.concatenate(
        (np.zeros((n,n)), np.eye(n,n)),
        axis=0,
    )

    T_mat = 100000.0 * np.eye(2*n, 2*n)
    M_mat = step * np.eye(2*n, 2*n)
    N_mat = step * np.eye(n, n)

    N = int((t_final - t_start) / step)
    print('Backward calculations')
    P_mat_list = np.zeros((N+1, 2*n, 2*n))
    P_mat_list[N] = T_mat
    for i in tqdm(range(N, 0, -1)):
        A_mat_T = np.transpose(A_mat)
        B_mat_T = np.transpose(B_mat)
        first = A_mat_T.dot(P_mat_list[i]).dot(A_mat)
        second = N_mat + B_mat_T.dot(P_mat_list[i]).dot(B_mat)
        second = np.linalg.inv(second)
        second = A_mat_T.dot(P_mat_list[i]).dot(B_mat).dot(second).dot(B_mat_T).dot(P_mat_list[i]).dot(A_mat)
        P_mat_list[i-1] = M_mat + first - second

    print('Control and trajectory calculations')
    v = np.zeros((N, n))
    z = np.zeros((N+1, 2*n))
    z[0] = z_start - z_final
    for i in tqdm(range(N)):
        B_mat_T = np.transpose(B_mat)
        tmp = N_mat + B_mat_T.dot(P_mat_list[i]).dot(B_mat)
        tmp = np.linalg.inv(tmp)
        v[i] = - tmp.dot(B_mat_T).dot(P_mat_list[i]).dot(A_mat).dot(z[i])
        z[i+1] = A_mat.dot(z[i]) + B_mat.dot(v[i])
        

    print('Translate trajectory')
    for i in tqdm(range(N+1)):
        z[i] = z[i] + z_final

    if M is not None and L is not None:
        print('Translate control')
        for i in tqdm(range(N)):
            v[i] = M(model, z[i]).dot(v[i]) + L(model, z[i])

    return z, v