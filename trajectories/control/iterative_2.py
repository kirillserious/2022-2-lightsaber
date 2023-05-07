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

class Iterative2:
    def __init__(self,
        f_z: MatrixMultiFcn,
        f_u: MatrixMultiFcn,
        qf: Cost,
        q: Cost,
        r: Cost,
        step: float,
        rho:float=0, # Also can use 1e-8
        beta1:float=1e-8,
        beta2:float=100.0,
    ):
        self.f_z = f_z
        self.f_u = f_u
        self.qf = qf
        self.q = q
        self.r = r
        self.step = step
        self.rho = rho
        self.beta1 = beta1
        self.beta2 = beta2

    def __cost(self, z: Union[Matrix, List[Vector]], u:Matrix)->float:
        step = self.step
        if isinstance(z, list):
            N = len(z)
        else:
            N = z.shape[0]
        
        cost = 0.0
        for k in tqdm(range(0, N-2)):
            cost += self.q.cost(z[k], step)
            cost += self.r.cost(u[k], step)
        cost += self.qf.cost(z[N-1], step)

        return cost


    def __backward_pass(self, z_nominal:Union[Matrix, List[Vector]], u_nominal:Matrix)->Tuple[Matrix, Vector, float]:
        step = self.step

        if isinstance(z_nominal, list):
            N = len(z_nominal)
        else:
            N = z_nominal.shape[0]
        
        z_shape = z_nominal[0].shape[0]
        u_shape = u_nominal[0].shape[0]
        
        p_N = self.qf.d_cost(z_nominal[N-1], step)
        P_N = self.qf.dd_cost(z_nominal[N-1], step)
        
        while True:
            p = p_N
            P = P_N
            K = np.zeros((N-1, u_shape, z_shape))
            d = np.zeros((N-1, u_shape))
            delta_v_linear = 0.0
            delta_v_quadratic = 0.0
            print('Backward Pass Started')
            for k in tqdm(range(N-2, -1, -1)):
                f_z = self.f_z(z_nominal[k], u_nominal[k], step)
                f_u = self.f_u(z_nominal[k], u_nominal[k], step)
                q_z = self.q.d_cost(z_nominal[k], step)
                q_zz = self.q.dd_cost(z_nominal[k], step)
                r_u = self.r.d_cost(u_nominal[k], step)
                r_uu = self.r.dd_cost(u_nominal[k], step)

                Q_z = q_z + f_z.T.dot(p)
                Q_u = r_u + f_u.T.dot(p)
                Q_zz = q_zz + f_z.T.dot(P).dot(f_z)
                Q_uu = r_uu + f_u.T.dot(P).dot(f_u)
                # Some where here maybe need to check det(Q_uu + rho*I) and
                # continue cycle with increased rho.
                Q_uz = f_u.T.dot(P).dot(f_z)
                Q_zu = f_z.T.dot(P).dot(f_u)

                tmp = -np.linalg.inv(Q_uu + self.rho*np.eye(u_shape))
                K[k] = tmp.dot(Q_uz)
                d[k] = tmp.dot(Q_u)
                #print(d[k])
                p = Q_z + K[k].T.dot(Q_uu).dot(d[k]) + K[k].T.dot(Q_u) + Q_zu.dot(d[k])
                P = Q_zz + K[k].T.dot(Q_uu).dot(K[k]) + K[k].T.dot(Q_uz) + Q_zu.dot(d[k])

                delta_v_linear += d[k].T.dot(Q_u)
                delta_v_quadratic += d[k].T.dot(Q_uu).dot(d[k])/2.0
            break

        return K, d, delta_v_linear, delta_v_quadratic


    def __forward_pass(self, z_nominal, u_nominal, K, d, delta_v_linear, delta_v_quadratic):
        J_nominal = self.__cost(z_nominal, u_nominal)
        if isinstance(z_nominal, list):
            N = len(z_nominal)
        else:
            N = z_nominal.shape[0]
        
        z = np.zeros((N, z_nominal[0].shape[0]))
        z[0] = z_nominal[0]
        u = np.zeros((N-1, u_nominal[0].shape[0]))
        alpha = 1.0
        gamma = 2.0
        while True:
            for k in tqdm(range(N-2)):
                u[k] = u_nominal[k] + K[k].dot(z[k] - z_nominal[k]) + alpha * d[k]
                #print(u[k])
                # TODO: Use just f
                f_z = self.f_z(z_nominal[k], u_nominal[k], self.step)
                f_u = self.f_u(z_nominal[k], u_nominal[k], self.step)
                z[k+1] = z_nominal[k+1] + f_z.dot(z[k]-z_nominal[k]) + f_u.dot(u[k]-u_nominal[k])
            
            print("Check search condition")
            J = self.__cost(z, u)
            search = -(J_nominal - J)/(alpha * delta_v_linear + alpha**2 * delta_v_quadratic)
            print(J_nominal)
            print(J)
            print(J_nominal - J)
            print(alpha * delta_v_linear + alpha**2 * delta_v_quadratic)
            if (search < self.beta1 or search > self.beta2) and alpha > 0.1:
                print("Search condition failed. search=%.2f" % (search))
                alpha = alpha / gamma
                continue
            break
        print("J=%.2f" % (J))
        return z, u, J


    def improve(self, z_nominal:Union[Matrix, List[Vector]], u_nominal:Matrix)->Tuple[Matrix, Matrix]:        
        K, d, delta_v_linear, delta_v_quadratic = self.__backward_pass(z_nominal, u_nominal)
        z, u, J = self.__forward_pass(z_nominal, u_nominal, K, d, delta_v_linear, delta_v_quadratic)
        return u
