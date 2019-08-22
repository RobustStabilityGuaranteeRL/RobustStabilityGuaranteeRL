#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import numpy as np
import binascii
from time import sleep
 
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
DATA_LENTH = 6
a_num=0;
diretion_flag = True
direction_num = 0
num_arry = [0,0,0,0,0,0,0,0,0,0]

send_data = [0xff,0xde,0x03,0x00,0x00,0x00]

## Function:send data_arry
def DataSend(data_arry,data_lenth):
  for data_num in range(0,data_lenth):
    data_arry_byte = data_arry[data_num].to_bytes(length=1, byteorder='big')
    ser.write(data_arry_byte)
    #sleep(0.005)
   

def recv(serial):
        global data
        data = serial.read(1)
        return data

while True:
#-------------------------------------Send data------------------------------------------#
  if diretion_flag == True:
    send_data[2]=send_data[2]&0x07
    send_data[2]=send_data[2]|0x00
  elif diretion_flag == False:
    send_data[2]=send_data[2]&0x07
    send_data[2]=send_data[2]|0x80
    
  if direction_num >= 50:
    direction_num=0
    if diretion_flag == True:
      diretion_flag = False
    else: 
      diretion_flag = True
    
      
  DataSend(send_data,DATA_LENTH)
  sleep(0.01)
  direction_num = direction_num+1
  print('######################',direction_num,'++++',diretion_flag,'---',)
    
#-------------------------------------Recieve data--------------------------------------------#
  data = recv(ser)
  print(data)

  if data==0xff:
    a_num=0
  else: 
    num_arry[a_num]=ord(data)

  a_num=a_num+1
  if a_num>9:
    a_num =0

  print(num_arry)
  pos=num_arry[1]+num_arry[2]*256+num_arry[3]*65536
  angle = num_arry[4]+num_arry[5]*256

  print('the receive data is:%c',pos,angle)
 
