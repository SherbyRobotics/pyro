#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np
import matplotlib

from gro640_robots import DrillingRobot
from gro640_robots import DrillingRobotOnJig
from augn1301      import CustomDrillingController


# Model dynamique du robot
sys = DrillingRobotOnJig()

# Controller
model = DrillingRobot()
ctl   = CustomDrillingController( model ) # Empty do nothing controller template

# Closed-loop dynamic
clsys = ctl + sys

# États initiaux
#clsys.x0 =  np.array([0.7,1.45,-1.4,0,0,0]) #  Tombe dans le trou
clsys.x0 =  np.array([0,1.4,-1.3,0,0,0]) #

# Simulation
tf = 6
n = 100001
clsys.compute_trajectory( tf, n, 'euler')
clsys.plot_trajectory('x')
clsys.plot_trajectory('u')


# Exemple extraction des données pour analyse
t        = clsys.traj.t
q_traj   = clsys.traj.x[:,0:3]  # Trajectoire des angles du robot
dq_traj  = clsys.traj.x[:,3:6]  # Trajectoire des vitesses du robot
tau_traj = clsys.traj.u         # Trajectoire des couples du robot

# Force de contact
##########################################################################################
# NOTE: les impacts de sont pas modèlisés dans la simulation, le comportement 
# transitoire obtenu n'est pas représentatif de ce que serait le vrai comportement.
# Le comportement quasi-statique (lorsque les accélérations sont faibles) est fiable.
##########################################################################################

f_traj   = np.zeros((n,3))
for i in range(n):
    f_traj[i,:] = sys.f_ext( q_traj[i,:] , dq_traj[i,:] )

fig , plots = matplotlib.pyplot.subplots(3, sharex=True, figsize=(4, 3),
                              dpi=300, frameon=True)
plots[0].plot( t , f_traj[:,0] , 'b') 
plots[0].set_ylabel('F_x [N]', fontsize=5 )
plots[0].grid(True)
plots[0].tick_params( labelsize = 5 )
plots[1].plot( t , f_traj[:,1] , 'b') 
plots[1].set_ylabel('F_y [N]', fontsize=5 )
plots[1].grid(True)
plots[1].tick_params( labelsize = 5 )
plots[2].plot( t , f_traj[:,2] , 'b')
plots[2].set_ylabel('F_z [N]', fontsize=5 )
plots[2].grid(True)
plots[2].tick_params( labelsize = 5 )
plots[2].set_xlabel('Time [sec]', fontsize=5 )
fig.canvas.manager.set_window_title('Contact forces')

# End-effector position
clsys.plot_end_effector_trajectory()

# Robot animation
clsys.animate_simulation( is_3d = True )