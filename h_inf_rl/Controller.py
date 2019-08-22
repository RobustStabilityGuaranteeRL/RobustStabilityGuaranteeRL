#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import SerialFunction


Balance_KP = 80   #80
Balance_KD = 30   #30
Position_KP = 30  #300 75
Position_KD = 770  #770 150

theta = 1.0

desire_pos = 220000
desire_ang = 3150  #1024 #3150

current_pos = 0
current_pos_error = 0
last_pos_error = 0
pos_bias = 0
pos_control_num = 0
Motor_pos = 0

current_ang = 0
current_ang_error = 0
last_ang_error = 0

## Function: Contorl func
def Pos_Contorler(pos):
  global current_pos_error,last_pos_error,current_pos,pos_bias
  
  current_pos = pos
  current_pos_error =  current_pos - desire_pos
  pos_bias = pos_bias * 0.8
  pos_bias = pos_bias + 0.2 * current_pos_error
  
  motor_pos = Position_KP * pos_bias/100 + Position_KD * (pos_bias - last_pos_error)/100
  
  last_pos_error = pos_bias
#  print("motor_posmotor_posmotor_pos:",motor_pos/100)
  return motor_pos
  
def Ang_Contorler(ang):
  global current_ang,current_ang_error,last_ang_error
  current_ang = ang
  current_ang_error = current_ang - desire_ang 
  motor_ang = - Balance_KP * current_ang_error - Balance_KD * (current_ang_error - last_ang_error)
  last_ang_error = current_ang_error
  
  return motor_ang

def Contorller(Pos,Ang):
  global pos_control_num,Motor_pos,theta

  if pos_control_num > 4:
    Motor_pos = theta * Pos_Contorler(Pos)
    pos_control_num = 0
    
  pos_control_num = pos_control_num + 1
  
  Motor_ang = Ang_Contorler(Ang)
#  print("++++++++++++++++++++++++++++++++++++++",Motor_ang, Motor_pos)
  Motor = Motor_ang - Motor_pos
#  Motor = Motor_ang
  if Motor > 4500:
    Motor = 4500
  if Motor < -4500:
    Motor = -4500
  
  return Motor
  

#  print("len(send_data)",len(send_data))
  
#  print(send_data)
