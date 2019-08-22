#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import sys
import numpy as np
import binascii
from time import sleep
 
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)

DATA_LENTH = 1024

send_data = [0xde,0x01,0x00,0x00,0x00,0x00]

## Function:send data_arry
def DataSend(data_arry,data_size):
  for data_num in range(0,DATA_LENTH):
    data_arry_byte = data_arry[data_num].to_bytes(length=1, byteorder='big')
    ser.write(data_arry_byte)
    if data_num == data_size-1:
      break

## Function:recieve data_arry
def DataRecieve(data_size):
  arry_num =0
  data_arry = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  for reci_num in range(0,DATA_LENTH):
    data = ord(ser.read(1))
    if data == 0xff:
      arry_num =0
      
    else:
      arry_num = arry_num+1
    
    data_arry[arry_num]=data
    
    if arry_num == data_size-1:
      break
  return data_arry
  
## Function: analys the position and angle 
def Get_Pos_Ang():
  data_arry_PA = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  data_arry_PA = DataRecieve(len(data_arry_PA))
#  print("REC:",data_arry_PA)
  position = data_arry_PA[1]+data_arry_PA[2]*256+data_arry_PA[3]*65536
  
  angle = data_arry_PA[4]+data_arry_PA[5]*256
  
  omega = (data_arry_PA[6]&0x7f)+(data_arry_PA[7]&0x7f)*128
  if ((data_arry_PA[6]&0x80) == 0x80) or ((data_arry_PA[7]&0x80) == 0x80):
    omega = -omega
    
  pos_dot = (data_arry_PA[8]&0x7f)+(data_arry_PA[9]&0x7f)*128+(data_arry_PA[10]&0x7f)*16384
  if ((data_arry_PA[8]&0x80) == 0x80) or ((data_arry_PA[9]&0x80) == 0x80) or ((data_arry_PA[10]&0x80) == 0x80):
    pos_dot = -pos_dot
    
  return position,pos_dot,angle,omega
  
## Functoin: Data send 
def DataSendControl(controllermotor):
  global send_data
#  print("controllermotor",controllermotor)
  if abs(controllermotor)<128:
    send_data[2] = abs(controllermotor)&0x7f
    send_data[3] = 0x00
    if controllermotor<0:
      send_data[2] = send_data[2]|0x80
  else:
    send_data[2] = abs(controllermotor)&0x7f
    send_data[3] = (abs(controllermotor)>>7)&0x7f
    if controllermotor<0:
      send_data[2] = send_data[2]|0x80
      send_data[3] = send_data[3]|0x80
      
  DataSend(send_data,len(send_data))
  
## Function: 
def Unification(position,pos_dot,angle,omega):
  Pos_rel = (220000-position)/100000;
  Pos_dot = - pos_dot*40/100000;
  Ang_rel = angle * 2 * np.pi / 4096;
  Ang_dot = -omega * 40 *2 *np.pi / 4096;
  Ang_rel = -Ang_rel + 0.12545617403361722 + 1.5 * np.pi;
  return Pos_rel,Pos_dot,Ang_rel,Ang_dot
'''  
if __name__ == "__main__":
  while True:
    DataSend(send_data,len(send_data))
    sleep(0.02)
    Pos,Ang = Get_Pos_Ang(data_arry)
    print("Pos Ang",Pos,Ang )
'''
 
