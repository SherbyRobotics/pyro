# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np

from scipy.integrate import odeint


##########################################################################
# Trajectory 
##########################################################################

class Trajectory():
    """ Simulation data """

    _dict_keys = ['x', 'u', 't', 'dx', 'y', 'r', 'J', 'dJ']

    def __init__(self, x, u, t, dx, y, r=None, J=None, dJ=None):
        """
        x:  array of dim = ( time-steps , sys.n )
        u:  array of dim = ( time-steps , sys.m )
        t:  array of dim = ( time-steps , 1 )
        y:  array of dim = ( time-steps , sys.p )
        """

        self.x  = x
        self.u  = u
        self.t  = t
        self.dx = dx
        self.y  = y
        self.r  = r
        self.J  = J
        self.dJ = dJ

        self._compute_size()
        
        
    ############################
    def _asdict(self):
        
        return {k: getattr(self, k) for k in self._dict_keys}
    
    
    ############################
    def save2(self, name = 'trajectory' ):
        
        #TODO ne fonctionne pas cet version, voir avec Françis?
        np.savez(name , **self._asdict())
        
        
    ############################
    def save(self, name = 'trajectory.npy' ):
        
        data = np.array( [ self.x , 
                           self.u , 
                           self.t ,
                           self.dx,
                           self.y ,
                           self.r ,
                           self.J , 
                           self.dJ ] )
        
        np.save( name , data )
        
    
    ############################
    @classmethod
    def load(cls, name):
        try:
            # try to load as new format (np.savez)
            with np.load(name) as data:
                return cls(**data)

        except ValueError:
            # If that fails, try to load as "legacy" numpy object array
            print('\nLoading Legacy Numpy object .npy')
            data = np.load(name, allow_pickle=True)
            return cls(*data)
        
        
    ############################
    def _compute_size(self):
        
        #print(self.t)
        
        self.time_final = self.t.max()
        self.time_steps = self.t.size

        self.n = self.x.shape[1]
        self.m = self.u.shape[1]
        
        self.ubar = np.zeros( self.m )

        # Check consistency between signals
        for arr in [self.x, self.y, self.u, self.dx, self.r, self.J, self.dJ]:
            if (arr is not None) and (arr.shape[0] != self.time_steps):
                raise ValueError("Result arrays must have same length along axis 0")
                

    ############################
    def t2u(self, t ):
        """ get u from time """

        # Find time index
        i = (np.abs(self.t - t)).argmin()

        # Find associated control input
        u = self.u[i,:]
        
        #if t > self.time_final:
        #    u = self.ubar

        return u
    

    ############################
    def t2x(self, t ):
        """ get x from time """

        # Find time index
        i = (np.abs(self.t - t)).argmin()

        # Find associated state
        return self.x[i,:]
    


##########################################################################
# Simulator
##########################################################################

class Simulator:
    """Simulation Class for open-loop ContinuousDynamicalSystem

    Parameters
    -----------
    cds    : Instance of ContinuousDynamicSystem
    tf     : float : final time for simulation
    n      : int   : number of time steps
    solver : {'ode', 'euler'}
    """
    
    ############################
    def __init__(
        self, ContinuousDynamicSystem, tf=10, n=10001, solver='ode'):

        self.cds    = ContinuousDynamicSystem
        self.t0     = 0
        self.tf     = tf
        self.n      = int(n)
        self.dt     = ( tf + 0.0 - self.t0 ) / ( n - 1 )
        self.solver = solver
        self.x0     = self.cds.x0
        self.cf     = self.cds.cost_function 
        
        # Check Initial condition state-vector
        if self.x0.size != self.cds.n:
            raise ValueError(
                "Number of elements in x0 must be equal to number of states"
            )
            

    ##############################
    def compute(self):
        """ Integrate trought time """

        t  = np.linspace( self.t0 , self.tf , self.n )

        if self.solver == 'ode':

            x_sol = odeint( self.cds.fsim , self.x0 , t)

            # Compute inputs-output values
            y_sol  = np.zeros(( self.n , self.cds.p ))
            u_sol  = np.zeros((self.n,self.cds.m))
            dx_sol = np.zeros((self.n,self.cds.n))

            for i in range(self.n):
                ti = t[i]
                xi = x_sol[i,:]
                ui = self.cds.t2u( ti )

                dx_sol[i,:] = self.cds.f( xi , ui , ti )
                y_sol[i,:]  = self.cds.h( xi , ui , ti )
                u_sol[i,:]  = ui

        elif self.solver == 'euler':

            x_sol  = np.zeros((self.n,self.cds.n))
            dx_sol = np.zeros((self.n,self.cds.n))
            u_sol  = np.zeros((self.n,self.cds.m))
            y_sol  = np.zeros((self.n,self.cds.p))

            # Initial State
            x_sol[0,:] = self.x0
            dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
            for i in range(self.n):

                ti = t[i]
                xi = x_sol[i,:]
                ui = self.cds.t2u( ti )

                if i+1<self.n:
                    dx_sol[i]    = self.cds.f( xi , ui , ti )
                    x_sol[i+1,:] = dx_sol[i] * dt + xi

                y_sol[i,:] = self.cds.h( xi , ui , ti )
                u_sol[i,:] = ui
                
        #########################
        traj = Trajectory(
          x = x_sol,
          u = u_sol,
          t = t,
          dx= dx_sol,
          y = y_sol
        )
        #########################
        
        # Compute Cost function
        if self.cf is not None :
            traj = self.cf.trajectory_evaluation( traj )
        
        return traj



###############################################################################
# Closed Loop Simulator
###############################################################################
    
class CLosedLoopSimulator(Simulator):
    """ 
    Simulation Class for closed-loop ContinuousDynamicalSystem 
    --------------------------------------------------------
    CLSystem  : Instance of ClosedLoopSystem
    tf : final time
    n  : number of point
    solver : 'ode' or 'euler'
    --------------------------------------------------------
    Use this class instead of Simulation() in order to access
    internal control inputs
    """
    
    ############################
    def __init__(self, ClosedLoopSystem, tf=10, n=10001, solver='ode'):
        
        # Mother class init
        super().__init__(ClosedLoopSystem, tf, n, solver)
        
        # Special cases
        
        # Use the plant cost function for closed-loop sys
        self.plant_cf = ClosedLoopSystem.plant.cost_function
        

    ###########################################################################
    def compute(self):
        
        traj = Simulator.compute(self)
        
        u = self._compute_control_inputs( traj )

        cl_traj = Trajectory(
            x  = traj.x,
            u  = u,
            t  = traj.t,
            dx = traj.dx,
            y  = traj.y,
            r  = traj.u.copy() # reference is input of global sys
        )
        
        # Compute Cost function
        if self.plant_cf is not None :
            
            cl_traj = self.plant_cf.trajectory_evaluation( cl_traj )

        return cl_traj
        

    ###########################################################################
    def _compute_control_inputs(self, traj ):
        """ Compute internal control inputs of the closed-loop system """

        r = traj.u.copy() # reference is input of combined sys
        u = np.zeros((self.n,self.cds.plant.m))

        # Compute internal input signal_proc
        for i in range(self.n):

            ri = r[i,:]
            yi = traj.y[i,:]
            ti = traj.t[i]

            ui = self.cds.controller.c( yi , ri , ti )
            
            u[i,:] = ui

        return u
    
    
###############################################################################
# Dynamic Closed Loop Simulator
###############################################################################
    
class DynamicCLosedLoopSimulator( CLosedLoopSimulator ):
    """ 
    Specific simulator for extracting internal control signal
    """

    ###########################################################################
    def _compute_control_inputs(self, traj ):
        """ Compute internal control inputs of the closed-loop system """

        r = traj.u.copy() # reference is input of combined sys
        u = np.zeros((self.n,self.cds.plant.m))

        # Compute internal input signal_proc
        for i in range(self.n):

            ri = r[i,:]
            yi = traj.y[i,:]
            xi = traj.x[i,:]
            ti = traj.t[i]
            
            # extract internal controller states
            xi,zi = self.cds._split_states( xi ) 

            ui = self.cds.controller.c( zi, yi , ri , ti )
            
            u[i,:] = ui

        return u
