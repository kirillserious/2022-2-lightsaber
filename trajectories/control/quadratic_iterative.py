from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union, List

import numpy as np
from tqdm import tqdm

from common import Model, Vector, Matrix

# fcn(model, zk, uk, step)
VectorFcn = Callable[[Model, Vector, Vector, float], Vector]

@dataclass
class QuadraticIterative:
    model: Model
    fk_z: VectorFcn
    fk_u: VectorFcn
    z_final: Vector
    Q_final: Matrix
    Q: Matrix
    R: Matrix
    step:Optional[float]=0.001
    
    def improve(self, z_nominal:Union[Matrix, List[Vector]], u_nominal:Matrix)->Tuple[Matrix, Matrix]:
        model = self.model
        fk_z = self.fk_z
        fk_u = self.fk_u
        z_final = self.z_final
        Q_final = self.Q_final
        Q = self.Q
        R = self.R
        step = self.step

        if isinstance(z_nominal, list):
            N = len(z_nominal)
        else:
            N = z_nominal.shape[0]
        
        print('Calculation of S started')
        K = [None] * N
        K_u = [None] * N
        K_v = [None] * N
        v = [None] * N
        v[N-1] = np.matmul(Q_final, z_nominal[N-1] - z_final)

        Si = Q_final
        for i in tqdm(range(N-2, -1, -1)):
            ui = u_nominal[i]
            zi = z_nominal[i]

            Ai = fk_z(model, zi, ui, step)
            Bi = fk_u(model, zi, ui, step)
            Bi_T = np.transpose(Bi)
            Bi_T_Si = Bi_T.dot(Si)
            support = np.linalg.inv(Bi_T_Si.dot(Bi) + R)


            K[i] = support.dot(Bi_T_Si).dot(Ai)
            K_v[i] = support.dot(Bi_T)
            K_u[i] = support.dot(R)

            Ai_minus_Bi_K = Ai - Bi.dot(K[i])

            Si = np.transpose(Ai).dot(Si).dot(Ai_minus_Bi_K) + Q
            v[i] = np.matmul(np.transpose(Ai_minus_Bi_K), v[i+1]) - np.matmul(np.transpose(K[i]).dot(R), ui) + np.matmul(Q, zi)

            
        print('Calculation of control started')

        z = [None] * N
        z[0] = z_nominal[0]
        u = [None] * (N-1)
        delta_zi = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in tqdm(range(N-1)):
            delta_ui = -np.matmul(K[i], delta_zi) - np.matmul(K_v[i], v[i+1]) - np.matmul(K_u[i], u_nominal[i])
            # Experiment
            norm_delta_ui = np.linalg.norm(delta_ui)
            if norm_delta_ui > 1.0:
                delta_ui = delta_ui / norm_delta_ui
            # End of experiment
            u[i] = u_nominal[i] + delta_ui

            Ai = fk_z(model, z_nominal[i], u_nominal[i], step)
            Bi =  fk_u(model, z_nominal[i], u_nominal[i], step)
            delta_zi = np.matmul(Ai, delta_zi) + np.matmul(Bi, delta_ui)
            z[i+1] = z_nominal[i+1] + delta_zi
        
        return z, u

