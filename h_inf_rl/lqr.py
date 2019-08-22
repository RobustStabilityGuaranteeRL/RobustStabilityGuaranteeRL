import numpy as np
import math
import scipy.linalg as linalg
lqr = linalg.solve_continuous_are
import time
from collections import OrderedDict, deque
import os
from copy import deepcopy
import sys
#sys.path.append("..")
#import logger
#from variant import *


class LQR(object):
    def __init__(self):
        theta_threshold_radians = 20 * 2 * math.pi / 360
        length = 0.185
        masscart = 0.2275
        masspole = 0.0923
        total_mass = (masspole + masscart)
        polemass_length = (masspole * length)
        g = 10
        H = np.array([
            [1, 0, 0, 0],
            [0, total_mass, 0, - polemass_length],
            [0, 0, 1, 0],
            [0, - polemass_length, 0, (2 * length) ** 2 * masspole / 3]
        ])

        Hinv = np.linalg.inv(H)

        A = Hinv @ np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, - polemass_length * g, 0]
        ])
        B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))
        Q = np.diag([1/100, 0., 20 *(1/ theta_threshold_radians)**2, 0.])
        R = np.array([[0.1]])

        P = lqr(A, B, Q, R)
        Rinv = np.linalg.inv(R)
        K = Rinv @ B.T @ P
        # H = np.mat([[masscart+masspole, masspole*length],
        #             [masspole*length,   masspole*length**2]])
        # dev_G = np.mat([[0,0],
        #                 [0,-masspole*g*length]])
        # A_sup = np.concatenate([np.zeros([2,2]),
        #                         np.diag(np.ones([2]))], axis=1)
        # A_sub = np.concatenate([-H.I*dev_G,np.zeros([2,2])],axis=1)
        # A = np.concatenate([A_sup,A_sub],axis=0)
        # B_dev = np.mat([[0],
        #                 [1]])
        # B = np.concatenate([np.zeros([2,1]),
        #                     H.I*B_dev],axis=0)
        #
        # Q = np.diag([0.1, 1.0, 100.0, 5.0])
        #
        # R = np.mat([1.])
        # K = np.mat(np.ones([1,4]))
        # P = np.mat(np.zeros([4,4]))
        # P_piao = np.mat(np.diag(np.ones([4])))
        # i = 0
        # ### optimize the controler
        # while np.linalg.norm(P_piao-P)>1e-8:
        #     P = P_piao
        #     K = -(R+gamma*B.T*P*B).I*B.T*P*A
        #     P_piao = Q + K.T*R*K + gamma * (A+B*K).T * P *(A+B*K)
        #     i += 1

        self.K = K
        print("KKKKKKK",K)
        


if __name__=='__main__':
    lqr_policy = LQR()
