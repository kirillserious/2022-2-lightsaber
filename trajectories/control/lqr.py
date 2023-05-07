from typing import Optional, Tuple, Callable

import numpy as np
from tqdm import tqdm

from common import Model, Vector, Matrix

MatrixFcn = Callable[[Model, Vector], Matrix]
VectorFcn = Callable[[Model, Vector], Vector]

def LQR(
    model:Model,
    M: MatrixFcn,
    L: VectorFcn,
    z_start: Vector,
    t_start: float,
    z_final: Vector,
    t_final: float,
    step: Optional[float] = 0.001,
)->Matrix:
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
    
    w1 = step * 0.00001
    w2 = step * 0.00001
    Q_mat = np.concatenate(
        (np.concatenate((np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]), np.zeros((n,n))), axis=1), np.zeros((n, 2*n))),
        axis=0,
    )

    N = int((t_final - t_start) / step)
    print('Backward calculations')
    S_mat = np.zeros((N+1, 2*n, 2*n))
    v = np.zeros((N+1, 2*n))

    S_mat[N] = np.eye(2*n)
    v[N] = -z_final
    for k in tqdm(range(N, 0, -1)):
        K_inv = np.linalg.inv(np.eye(2*n) + B_mat.dot(B_mat.T).dot(S_mat[k])/w2)
        S_mat[k-1] = w1 * Q_mat + A_mat.T.dot(S_mat[k]).dot(K_inv).dot(A_mat)
        v[k-1] = A_mat.T.dot(np.eye(2*n) - S_mat[k].dot(K_inv).dot(B_mat).dot(B_mat.T)/w2).dot(v[k])

    print('Control and trajectory calculations')
    u = np.zeros((N, n))
    z = np.zeros((N+1, 2*n))
    z[0] = z_start
    for k in tqdm(range(N)):
        K_inv = np.linalg.inv(np.eye(2*n) + B_mat.dot(B_mat.T).dot(S_mat[k+1])/w2)
        z[k+1] = K_inv.dot(A_mat.dot(z[k]) - B_mat.dot(B_mat.T).dot(v[k+1])/w2)
        lambda_k_plus_1 = S_mat[k+1].dot(z[k+1]) + v[k+1]

        u[k] = - B_mat.T.dot(lambda_k_plus_1)/w2

    if M is not None and L is not None:
        print('Translate control')
        for i in tqdm(range(N)):
            u[i] = M(model, z[i]).dot(u[i]) + L(model, z[i])

    return u