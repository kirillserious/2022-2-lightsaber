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
class IterativeDDP:
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
        qf = self.qf
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

        lambdas = np.zeros((N, z_nominal[0].shape[0]))
        print(z_nominal[0].shape[0])
        print(qf_z(model, z_nominal[N-1], step))
        lambdas[N-1] = - qf_z(model, z_nominal[N-1], step)
        u = np.zeros((N-1, u_nominal[0].shape[0]))
        eta = 0.01
        for k in tqdm(range(N-2, -1, -1)):
            lambdas[k] = f_z(model, z_nominal[k], u_nominal[k], step).dot(lambdas[k+1]) - q_z(model, z_nominal[k], step)
            delta_u = r_u(model, u_nominal[k], step) - f_u(model, z_nominal[k], u_nominal[k], step).T.dot(lambdas[k])
            u[k] = u_nominal[k] - eta * delta_u

        return u

