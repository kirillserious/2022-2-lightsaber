from typing import Optional, Callable, Tuple, Union, List
from cost import Cost
import numpy as np
from tqdm import tqdm

from common import Model, Vector, Matrix

# fcn(model, zk, uk, step)
MatrixMultiFcn = Callable[[Vector, Vector, float], Matrix]

# fcn(model, zk or uk, step)
ScalarFcn = Callable[[Model, Vector, float], float]
VectorFcn = Callable[[Model, Vector, float], Vector]
MatrixFcn = Callable[[Model, Vector, float], Matrix]

class Iterative3:
    def __init__(self, f_z: MatrixMultiFcn, f_u: MatrixMultiFcn, qf: Cost, q: Cost, r: Cost, step: float, max_d:float = 50.0):
        self.f_z = f_z
        self.f_u = f_u
        self.qf = qf
        self.q = q
        self.r = r
        self.step = step
        self.max_d = max_d

    def improve(self, z_nominal:Union[Matrix, List[Vector]], u_nominal:Matrix)->Tuple[Matrix, Matrix]:
        step = self.step

        if isinstance(z_nominal, list):
            N = len(z_nominal)
        else:
            N = z_nominal.shape[0]

        S = np.zeros((N, z_nominal[0].shape[0], z_nominal[0].shape[0]))
        v = np.zeros((N, z_nominal[0].shape[0]))
        
        print('Start S,v backward computation')
        S[N-1] = self.qf.dd_cost(z_nominal[N-1], step)
        v[N-1] = self.qf.d_cost(z_nominal[N-1], step)

        for k in tqdm(range(N-2, -1, -1)):
            uk_nominal = u_nominal[k]
            zk_nominal = z_nominal[k]

            fk_z = self.f_z(zk_nominal, uk_nominal, step)
            fk_u = self.f_u(zk_nominal, uk_nominal, step)
            qk = self.q.cost(zk_nominal, step)
            qk_z = self.q.d_cost(zk_nominal, step)
            qk_zz = self.q.dd_cost(zk_nominal, step)
            rk = self.r.cost(uk_nominal, step)
            rk_u = self.r.d_cost(uk_nominal, step)
            rk_uu = self.r.dd_cost(uk_nominal, step)
            rk_uu_inv = np.linalg.inv(rk_uu)
            
            A_11 = fk_z
            A_12 = -fk_u.dot(rk_uu_inv).dot(fk_u.T)
            A_21 = qk_zz
            A_22 = fk_z.T
            B_1 = -fk_u.dot(rk_uu_inv).dot(rk_u)
            B_2 = qk_z

            #print(np.linalg.det(np.eye(z_nominal[0].shape[0])-A_12.dot(S[k+1])))
            S[k] = A_21 + A_22.dot(
                S[k+1]
            ).dot(
                np.linalg.inv(
                    np.eye(z_nominal[0].shape[0])-A_12.dot(S[k+1])
                )).dot(A_11)
            v[k] = A_22.dot(S[k+1]).dot(np.linalg.inv(np.eye(z_nominal[0].shape[0])-A_12.dot(S[k+1]))).dot(A_12.dot(v[k+1]) + B_1) + A_22.dot(v[k+1]) + B_2
    
        print('Start u forward computation')
        u = np.zeros((N-1, u_nominal[0].shape[0]))

        delta_z = np.zeros((z_nominal[0].shape[0]))
        max_norm_d = 0.0
        for k in tqdm(range(0, N-1)):
            uk_nominal = u_nominal[k]
            zk_nominal = z_nominal[k]

            fk_z = self.f_z(zk_nominal, uk_nominal, step)
            fk_u = self.f_u(zk_nominal, uk_nominal, step)
            rk_u = self.r.d_cost(uk_nominal, step)
            rk_uu = self.r.dd_cost(uk_nominal, step)
            rk_uu_inv = np.linalg.inv(rk_uu)

            K = -np.linalg.inv(rk_uu + fk_u.T.dot(S[k+1]).dot(fk_u)).dot(fk_u.T.dot(S[k+1].dot(fk_z)))
            d = -np.linalg.inv(rk_uu + fk_u.T.dot(S[k+1]).dot(fk_u)).dot(fk_u.T.dot(v[k+1]) + rk_u)
            
            norm_d = np.linalg.norm(d)
            if norm_d > max_norm_d:
                max_norm_d = norm_d
            
            
            delta_u = K.dot(delta_z) + d

            delta_z = fk_z.dot(delta_z) + fk_u.dot(delta_u)
            u[k] = delta_u + uk_nominal

        eps = self.max_d
        if max_norm_d < eps:
            return u

        print('%.02f' % (max_norm_d))
        delta_z = np.zeros((z_nominal[0].shape[0]))
        for k in tqdm(range(0, N-1)):
            uk_nominal = u_nominal[k]
            zk_nominal = z_nominal[k]

            fk_z = self.f_z(zk_nominal, uk_nominal, step)
            fk_u = self.f_u(zk_nominal, uk_nominal, step)
            rk_u = self.r.d_cost(uk_nominal, step)
            rk_uu = self.r.dd_cost(uk_nominal, step)
            rk_uu_inv = np.linalg.inv(rk_uu)

            K = -np.linalg.inv(rk_uu + fk_u.T.dot(S[k+1]).dot(fk_u)).dot(fk_u.T.dot(S[k+1].dot(fk_z)))
            d = -np.linalg.inv(rk_uu + fk_u.T.dot(S[k+1]).dot(fk_u)).dot(fk_u.T.dot(v[k+1]) + rk_u)
            d = eps * d / max_norm_d
            
            delta_u = K.dot(delta_z) + d

            delta_z = fk_z.dot(delta_z) + fk_u.dot(delta_u)
            u[k] = delta_u + uk_nominal
        
        return u

