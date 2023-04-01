from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union, List

import numpy as np
from tqdm import tqdm

from common import Model, Vector, Matrix

# fcn(model, zk, uk, step)
MatrixMultiFcn = Callable[[Model, Vector, Vector, float], Matrix]

# fcn(model, zk or uk, step)
ScalarFcn = Callable[[Model, Vector, float], float]
VectorFcn = Callable[[Model, Vector, float], Vector]
MatrixFcn = Callable[[Model, Vector, float], Matrix]

@dataclass
class Iterative:
    model: Model
    f_z: MatrixMultiFcn
    f_u: MatrixMultiFcn

    # Tailor decomposition of integral x constraints
    qf: ScalarFcn
    qf_z: VectorFcn
    qf_zz: MatrixFcn
    
    q: ScalarFcn
    q_z: VectorFcn
    q_zz: MatrixFcn

    r: ScalarFcn
    r_u: VectorFcn
    r_uu: MatrixFcn

    step:Optional[float]=0.001

    def improve(self, z_nominal:Union[Matrix, List[Vector]], u_nominal:Matrix)->Tuple[Matrix, Matrix]:
        model = self.model
        f_z = self.f_z
        f_u = self.f_u
        qf_z = self.qf_z
        qf_zz = self.qf_zz
        q = self.q
        q_z = self.q_z
        q_zz = self.q_zz
        r = self.r
        r_u = self.r_u
        r_uu = self.r_uu
        step = self.step

        if isinstance(z_nominal, list):
            N = len(z_nominal)
        else:
            N = z_nominal.shape[0]

        S = np.zeros((N, z_nominal[0].shape[0], z_nominal[0].shape[0]))
        v = np.zeros((N, z_nominal[0].shape[0]))
        
        print('Start S,v backward computation')
        S[N-1] = qf_zz(model, z_nominal[N-1], step)
        v[N-1] = qf_z(model, z_nominal[N-1], step)
        
        for k in tqdm(range(N-2, -1, -1)):
            uk_nominal = u_nominal[k]
            zk_nominal = z_nominal[k]

            fk_z = f_z(model, zk_nominal, uk_nominal, step)
            fk_u = f_u(model, zk_nominal, uk_nominal, step)
            qk = q(model, zk_nominal, step)
            qk_z = q_z(model, zk_nominal, step)
            qk_zz = q_zz(model, zk_nominal, step)
            rk = r(model, uk_nominal, step)
            rk_u = r_u(model, uk_nominal, step)
            rk_uu = r_uu(model, uk_nominal, step)
            rk_uu_inv = np.linalg.inv(rk_uu)
            
            A_11 = fk_z
            A_12 = -fk_u.dot(rk_uu_inv).dot(fk_u.T)
            A_21 = qk_zz
            A_22 = fk_z.T
            B_1 = -fk_u.dot(rk_uu_inv).dot(rk_u)
            B_2 = qk_z

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
        for k in tqdm(range(0, N-1)):
            uk_nominal = u_nominal[k]
            zk_nominal = z_nominal[k]

            fk_z = f_z(model, zk_nominal, uk_nominal, step)
            fk_u = f_u(model, zk_nominal, uk_nominal, step)
            rk_u = r_u(model, uk_nominal, step)
            rk_uu = r_uu(model, uk_nominal, step)
            rk_uu_inv = np.linalg.inv(rk_uu)

            delta_u = -rk_uu_inv.dot(rk_u + fk_u.T.dot(S[k+1].dot(delta_z) + v[k+1]))

            # Experiment
            norm_delta_u = np.linalg.norm(delta_u)
            if norm_delta_u > 1.0:
                delta_u = delta_u / norm_delta_u
            # End of experiment
            delta_z = fk_z.dot(delta_z) + fk_u.dot(delta_u)
            u[k] = delta_u + uk_nominal
        
        #exit()
        return u

