# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

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
    
    ############################
    def copy(self):
        """ copy object """
        
        x  = self.x.copy()
        u  = self.u.copy()
        t  = self.t.copy()
        dx = self.dx.copy()
        y  = self.y.copy()
        
        if self.r is not None:
            r  = self.r.copy()
        else:
            r = None
        
        if self.J is not None:
            J  = self.J.copy()
            dJ = self.dJ.copy()
        else:
            J  = None
            dJ = None

        new_traj = Trajectory(x, u, t, dx, y, r, J, dJ)
        
        return new_traj
    
    
    ###########################
    def generate_interpol_functions(self):
        """ """
        
        # Create interpol functions
        self.inter_t2x   = interp1d(self.t,self.x.T)
        self.inter_t2u   = interp1d(self.t,self.u.T)
        self.inter_t2dx  = interp1d(self.t,self.dx.T)
        self.inter_t2y   = interp1d(self.t,self.y.T)
        
        if self.r is not None:
            self.inter_t2r = interp1d(self.t,self.r.T)
            
        if self.J is not None:
            self.inter_t2J  = interp1d(self.t,self.J)
            self.inter_t2dJ = interp1d(self.t,self.dJ)
    
    
    ############################
    def re_sample(self, n ):
        """ Return new traj with interpolated new time vector """
        
        self.generate_interpol_functions()
        
        ti = self.t[0]
        tf = self.t[-1]
        
        t = np.linspace( ti, tf, n)
        
        x  = np.zeros(( n, self.x.shape[1] ))
        u  = np.zeros(( n, self.u.shape[1] ))
        dx = np.zeros(( n, self.dx.shape[1] ))
        y  = np.zeros(( n, self.y.shape[1] ))
        #r  = np.zeros(( n, self.r.shape[1] ))
        
        for i in range(n):
            
            x[i,:]  = self.inter_t2x(  t[i] )
            u[i,:]  = self.inter_t2u(  t[i] )
            dx[i,:] = self.inter_t2dx( t[i] )
            y[i,:]  = self.inter_t2y(  t[i] )
            #r[i,:]  = self.inter_t2r( t[i] )


        new_traj = Trajectory(x, u, t, dx, y)
        
        return new_traj
    


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
        If ode, uses `scipy.integrate.solve_ivp`. `euler` uses built-in
        solver based on the Euler method.
    """

    ############################
    def __init__(
        self, ContinuousDynamicSystem, tf=10, n=None, solver='ode'):

        self.cds    = ContinuousDynamicSystem
        self.t0     = 0
        self.tf     = tf
        self.n      = n
        self.solver = solver
        self.x0     = self.cds.x0
        self.cf     = self.cds.cost_function 

        # Check Initial condition state-vector
        if self.x0.size != self.cds.n:
            raise ValueError(
                "Number of elements in x0 must be equal to number of states"
            )


    ##############################
    def compute(self, **solver_args):
        """Integrate trough time

        Parameters
        -----------
        kwargs: Keyword arguments passed through to the solver (e.g. solve_ivp)
        """

        if self.solver == 'ode':
            t_eval = None
            if self.n is not None:
                t_eval = np.linspace(self.t0 , self.tf , int(self.n))

            # solve_ivp takes arguments in reverse order of fsim
            def solverfun(t, y):
                dy = self.cds.fsim(y, t)
                return dy

            sol = solve_ivp(
                solverfun,
                t_span=[self.t0, self.tf],
                y0=self.x0,
                t_eval=t_eval,
                **solver_args
            )

            # Compute inputs-output values
            t_sol = sol.t
            n_sol = t_sol.shape[0]
            x_sol = sol.y.transpose()
            y_sol  = np.zeros((n_sol, self.cds.p ))
            u_sol  = np.zeros((n_sol, self.cds.m))
            dx_sol = np.zeros((n_sol, self.cds.n))

            for i in range(n_sol):
                ti = t_sol[i]
                xi = x_sol[i, :]
                ui = self.cds.t2u( ti )

                dx_sol[i,:] = self.cds.f( xi , ui , ti )
                y_sol[i,:]  = self.cds.h( xi , ui , ti )
                u_sol[i,:]  = ui

        elif self.solver == 'euler':
            npts = 10001 if self.n is None else int(self.n)
            t_sol = np.linspace(self.t0 , self.tf , npts)

            x_sol  = np.zeros((npts, self.cds.n))
            dx_sol = np.zeros((npts, self.cds.n))
            u_sol  = np.zeros((npts, self.cds.m))
            y_sol  = np.zeros((npts, self.cds.p))

            # Initial State
            x_sol[0,:] = self.x0
            dt = ( self.tf + 0.0 - self.t0 ) / ( npts - 1 )
            for i in range(npts):

                ti = t_sol[i]
                xi = x_sol[i,:]
                ui = self.cds.t2u( ti )

                if i+1 < npts:
                    dx_sol[i]    = self.cds.f( xi , ui , ti )
                    x_sol[i+1,:] = dx_sol[i] * dt + xi

                y_sol[i,:] = self.cds.h( xi , ui , ti )
                u_sol[i,:] = ui
                
        #########################
        traj = Trajectory(
          x = x_sol,
          u = u_sol,
          t = t_sol,
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
        Simulator.__init__(self, ClosedLoopSystem, tf, n, solver)
        
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
    def _compute_control_inputs(self, traj):
        """ Compute internal control inputs of the closed-loop system """

        r = traj.u.copy() # reference is input of combined sys
        npts = traj.t.shape[0]
        u = np.zeros([npts, self.cds.plant.m])

        # Compute internal input
        for i in range(npts):

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
        npts = traj.t.shape[0]
        u = np.zeros([npts ,self.cds.plant.m])

        # Compute internal input signal_proc
        for i in range(npts):

            ri = r[i,:]
            yi = traj.y[i,:]
            xi = traj.x[i,:]
            ti = traj.t[i]

            # extract internal controller states
            xi,zi = self.cds._split_states( xi ) 

            ui = self.cds.controller.c( zi, yi , ri , ti )

            u[i,:] = ui

        return u
