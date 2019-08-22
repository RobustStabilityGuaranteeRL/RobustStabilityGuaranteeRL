#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import SerialFunction
import Controller
from time import sleep
import LqrController
  
if __name__ == "__main__":
  contorl_motor = 0
  while True:
    ## get Position [200,000 : 220,000]  Angle[0 : 4096]  Omege [Angle per 25ms ]
    Pos,Pos_dot,Ang,Omega= SerialFunction.Get_Pos_Ang()
#    position,pos_dot,angle,omega
#      Pos = 230000
#      Ang = 3049
    ZeroPos = 220000
    
    print("Current Pos And Ang are:",Pos,Ang)
    print("*******************::::",Omega)
    
#    Pos_rel,Pos_dot,Ang_rel,Ang_dot = SerialFunction.Unification(Pos,Pos_dot,Ang,Omega)
    
#    print("The unificaiton parameters :  ",Pos_rel,Pos_dot,Ang_rel,Ang_dot)
    contorl_motor = Controller.Contorller(Pos,Ang,ZeroPos)

#    contorl_motor = LqrController.LqrControl(Pos,Ang)
    
    print("Controller OutPut:",contorl_motor)
    
    contorl_motor = round(contorl_motor) #get the 
    
    SerialFunction.DataSendControl(contorl_motor)
    sleep(0.001)
      
      
      
      
      
      
      

