#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#K = np.array([[-7.6028,-1.4675,-0.1,-0.2667]])
K = np.array([[-38.21702982,-1.77654387,-0.31622777,-1.61843025]])

des_Pos = 220000
des_Ang = 3150

pos_last = 0
ang_last = 0
Motor  = 0

def LqrControl(Pos,Ang):
  global pos_last,ang_last,Motor
  
  current_pos = Pos - des_Pos
  current_ang = Ang - des_Ang
  
  Diff_Pos = current_pos - pos_last
  Diff_Ang = current_ang - ang_last
  
  pos_last = current_pos
  ang_last = current_ang
  
  X_state = np.array([[current_ang,Diff_Ang,current_pos,Diff_Pos]])
  
  Motor = K[0][0] * X_state[0][0] +  K[0][1] * X_state[0][1] + K[0][2] * X_state[0][2] + K[0][3] * X_state[0][3]
  
  if Motor > 4500:
    Motor = 4500
    
  if Motor < -4500:
    Motor = -4500
  else:
    Motor = int(Motor)
  print("Motor",Motor)
  
  return Motor
