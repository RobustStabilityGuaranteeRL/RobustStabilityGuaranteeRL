#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
from . import SerialFunction
from . import Controller
from time import sleep
from . import LqrController
import numpy as np

K = 0.508

## motor parameter
V_s = 24   # nominal voltage
I_s = 7    # stall current
R = V_s/I_s    ## Resistance

I_n = 2.3 ## Nominal current
T_n = 2.6 * 9.8/100  # Nominal torque  N*M

k_t = T_n / I_n

I_no_load = 0.3   ## No load current
w_no_load = 1690* 2 *np.pi/ 60  ## rad/s no load speed
k_i = (V_s - I_no_load * R) / w_no_load
r = 1.75e-2  # radial of motor m
reduce_speed_ratio = 5.18

theta_threshold_radians = 60 * 2 * np.pi / 360
if __name__ == "__main__":
  contorl_motor = 0
  while True:
    ## get Position [200,000 : 220,000]  Angle[0 : 4096]  Omege [Angle per 25ms ]
    Pos,Ang,Omega = SerialFunction.Get_Pos_Ang()
#      Pos = 230000
#      Ang = 3049
    ZeroPos = 220000
    
    print("Current Pos And Ang are:",Pos,Ang)
    print("*******************::::",Omega)
    
#    contorl_motor = Controller.Contorller(Pos,Ang,ZeroPos)

    contorl_motor = LqrController.LqrControl(Pos,Ang)
    
    print("Controller OutPut:",contorl_motor)
    
    contorl_motor = round(contorl_motor) #get the 
    
    SerialFunction.DataSendControl(contorl_motor)
    sleep(0.001)

def read_sensor():
  Pos, Pos_dot, Ang, Omega = SerialFunction.Get_Pos_Ang()
  Pos, Pos_dot, Ang, Omega = SerialFunction.Unification(Pos, Pos_dot, Ang, Omega)

  return np.array([Pos, Pos_dot, Ang, Omega])

def transform_action(action, s):

  # control_motor = 1/K * 7200./12. * np.sign(action[0]) * np.sqrt(np.abs(action[0]))

  motor_speed = s[1]/ r * reduce_speed_ratio
  desired_torque = action * r
  desired_voltage = R * desired_torque/ k_t + k_i * motor_speed
  control_motor = desired_voltage /12 * 7200
  return control_motor

def step(action, s):

  contorl_motor = int(round(action[0]))
  print(contorl_motor)
  SerialFunction.DataSendControl(contorl_motor)
  s = read_sensor()
  Pos, Pos_dot, Ang, Omega = s
  cost = 1 * Pos ** 2 / 0.2 ** 2 + 20 * (Ang / theta_threshold_radians) ** 2
  return s, cost
      
      

