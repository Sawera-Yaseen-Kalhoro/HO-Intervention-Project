import numpy as np
# from math import cos,sin
from nav_utils.Pose import Pose3D

'''
    Pure Python implementation of the Extended Kalman Filter (EKF) for localization,
    using wheel encoders for Prediction, and IMU magnetometer yaw reading for Update.

    Note: Most functions are copied from PR lab.
'''

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

class EKF:
    def __init__(self,x0,P0) -> None:
        self.xk = x0
        self.Pk = P0

    def Prediction(self,xk_1,Pk_1,uk,Qk):
        self.xk_bar = self.f(xk_1,uk)
        Ak = self.Jfx(xk_1,uk)
        Wk = self.Jfw(xk_1)
        self.Pk_bar = Ak@Pk_1@Ak.T + Wk@Qk@Wk.T

        self.xk_bar[2,0] = wrap_angle(self.xk_bar[2,0])

        return self.xk_bar, self.Pk_bar

    def Update(self,xk_bar,Pk_bar,zk,Rk,Hk,Vk):

        Kk = Pk_bar @ Hk.T @ np.linalg.inv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T)
        innovation = zk-self.h(xk_bar)
        innovation[0,0] = wrap_angle(innovation[0,0])
        self.xk = xk_bar + Kk@(innovation)

        B = np.identity(np.shape(Kk@Hk)[0]) - Kk@Hk
        self.Pk = B@Pk_bar@B.T

        self.xk[2,0] = wrap_angle(self.xk[2,0])

        return self.xk,self.Pk

    def f(self,xk_1,uk):
        xk_bar = Pose3D(xk_1).oplus(uk)

        return xk_bar

    def Jfx(self,xk_1,uk):
        J = Pose3D(xk_1).J_1oplus(uk)

        return J

    def Jfw(self,xk_1):
        J = Pose3D(xk_1).J_2oplus()

        return J
    
    def h(self, xk):  # return the expected observations
        z = xk[2]

        return z  
    