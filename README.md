The latest version of this code is now hosted here: https://github.com/SherbyRobotics/pyro

![example workflow](https://github.com/SherbyRobotics/pyro/actions/workflows/python-app.yml/badge.svg)

# Pyro 

An object-based toolbox for robot dynamic simulation, analysis, control and planning. 

### A collection of dynamic systems:
<table>
  <tr>
    <th>
    <img src="https://www.alexandregirard.com/IMG/pyro/rocket.gif" alt="rocket" width="380"/>
    </th>
    <th>
    <img src="https://www.alexandregirard.com/IMG/pyro/3dofarm.gif" alt="cartpole" width="380"/>
    </th> 
  </tr>
  <tr>
    <td>
      <img src="https://www.alexandregirard.com/IMG/pyro/cartpole.gif" alt="cartpole_swing_up" width="380"/>
    </td>
    <td>
      <img src="https://www.alexandregirard.com/IMG/pyro/suspension.gif" alt="mass-spring" width="380"/>
    </td> 
  </tr>
</table>

### A collection of controller synthesis and planning tools:
<table>
  <tr>
    <th>
      Computed torque controller
      <img src="https://www.alexandregirard.com/IMG/pyro/ctc.png" alt="ctc" width="360"/>
    </th>
    <th>
      Sliding mode controller
      <img src="https://www.alexandregirard.com/IMG/pyro/smc.png" alt="smc" width="360"/>
    </th> 
  </tr>
  <tr>
    <th>
      Dynamic programming
      <img src="https://www.alexandregirard.com/IMG/pyro/J.gif" alt="cost2go" width="360"/>
    </th>
    <th>
      Optimal torque policy
      <img src="https://www.alexandregirard.com/IMG/pyro/pi.gif" alt="policy" width="360"/>
    </th> 
  </tr>
  <tr>
    <th>
      Rapidly-exploring random tree planning
      <img src="https://www.alexandregirard.com/IMG/pyro/rrt.png" alt="rrt" width="360"/>
    </th>
    <th>
      Direct collocation trajectory optimisation
      <img src="https://www.alexandregirard.com/IMG/pyro/doublependulum.gif" alt="doublependulum" width="360"/>
    </th> 
  </tr>
</table>

### A collection of analysis tools:
<table>
  <tr>
    <th>
      Simulation (computing trajectories)
      <img src="https://www.alexandregirard.com/IMG/pyro/traj.png" alt="traj" width="360"/>
    </th>
    <th>
      Phase plane analysis
      <img src="https://www.alexandregirard.com/IMG/pyro/pp.png" alt="phase-plane" width="360"/>
    </th> 
  </tr>
    <tr>
    <th>
      Generating animated simulations
      <img src="https://www.alexandregirard.com/IMG/pyro/plane.gif" alt="ani" width="360"/>
    </th>
    <th>
      Robot arm manipulability ellipsoid
      <img src="https://www.alexandregirard.com/IMG/pyro/elp.png" alt="elp" width="360"/>
  </tr>
  <tr>
    <th>
      Bode plot or output/input
      <img src="https://www.alexandregirard.com/IMG/pyro/bode.png" alt="bode" width="360"/>
    </th>
    <th>
      Pole zero map of output/input
      <img src="https://www.alexandregirard.com/IMG/pyro/pz.png" alt="pz" width="360"/>
  </tr>
  <tr>
    <th>
      Modal analysis (mode 1)
      <img src="https://www.alexandregirard.com/IMG/pyro/mode1.gif" alt="mode1" width="360"/>
    </th>
    <th>
      Modal analysis (mode 2)
      <img src="https://www.alexandregirard.com/IMG/pyro/mode2.gif" alt="mode2" width="360"/>
  </tr>
</table>

### Unified by a standardized *dynamic system*, *controller* and *planner* classes hierarchy

The concept of this toolbox is a hierachy of *dynamic system* objects, from the most generic representation (any non-linear differential equations) to more system specific representations such as mechanical system (second order equations), linear state space, manipulator equations, etc. This structure is then leveraged by analysis tools, from generic tools that work for all sub-class of dynamic systems such as running simulation and phase-plane analysis, to system-specific tools that leverage specific system propreties such as modal analysis for linear sub-class:

<img width="800" src="https://www.alexandregirard.com/IMG/pyro/classes.png" class="center">

The core of the library is a mother *dynamic system* class defined by a differential equation $\dot{x} = f(x,u,t)$, and optionnaly an output equation $y = h(x,u,t)$ and a foward kinematic equation that is used for generating animations:

<img width="500" src="https://www.alexandregirard.com/IMG/pyro/dynamic.png" class="center">

# How to use #

To learn how to use pyro, see the following notebook tutorials hosted on colab:

1.   [The Dynamic System class and basic functionnality](https://colab.research.google.com/drive/18eEL-n-dv9JZz732nFCMtqMThDcfD2Pr?usp=sharing)
2.   [Creating a custom dynamic class](https://colab.research.google.com/drive/1ILfRpL1zgiQZBOxwtbbpe0nl2znvzdWl?usp=sharing)
3.   [Closed-loop system and controllers objects](https://colab.research.google.com/drive/1mog1HAFN2NFEdw6sPudzW2OaTk_li0Vx?usp=sharing)
4.   The Linear System class (comin soon..)
4.   The Mechanical System class (coming soon..)
5.   [The Manipulator Robot class](https://colab.research.google.com/drive/1OILAhXRxM1r5PEB1BWaYtbR147Ff3gr1?usp=sharing)

Also see exemples scripts in pyro/examples/ 

and tutorial video (in french) on youtube: [here](https://youtube.com/playlist?list=PL6adNeJ0A8UtD_HQJTv6FHalnj0t7RyVa&si=SoZGOcYpsjLWvw27)


# Installation #

### Dependencies ####
Pyro is built only using core python librairies: 
* numpy
* scipy
* matplotlib

### Using in Colab ###

```
!git clone https://github.com/SherbyRobotics/pyro
import sys
sys.path.append('/content/pyro')
import pyro
```

### Using with Anaconda and Spyder IDE ###

#### 1. Download anaconda python distribution 
Download anaconda (including spyder IDE) available here: https://www.anaconda.com/products/individual

#### 2. Dowload pyro source code.
option a) Using git to clone the repo:
```bash
git clone https://github.com/SherbyRobotics/pyro.git
```
in the folder of your choice.

option b) Download the .zip using the Code/Download Zip link at the top of this page, and then unzip in the folder of your choice.

#### 3. Add the pyro folder to the pythonpath 
option a) [Easy spyder IDE only] Add it this the spyder menu at python/PYTHONPATH manager. 

In order to run pyro in the terminal directly of in another IDE like VS code, option b) or c) should be used.

option b) [conda]
```bash
conda develop /PATH/TO/PYRO
```

option c) [pip]
Go to the root directory of the pyro folder and run:
```bash
python -m pip install -e .
```

##### Graphical backend debuging 

By default pyro will try to use matplotlib Qt5Agg backend and interactive mode. You can modify the default graphical behavior by modifying the headers of the file pyro/analysis/graphical.py
In spyder IDE, you cand also change the graphics backend in the menu at python/Preferences/IPython console/Backend. Inline does not allow animations, it is best to use Automatic (for Windows and Ubuntu) or OS X (for Mac).


# Pyro tools list #

### Dynamic objects ###

- Continuous Dynamic system : $\dot{x} = f(x,u)$
- Linear System : $\dot{x} = A x + B u $
  - Transfer function 
  - Exemples: mass-spring-damper
- Mechanical System : $H(q)\ddot{q} + C(\dot{q},q)\dot{q} = \sum F $
  - Manipulator Robot : $\dot{r} = J(q) \dot{q}$
    - Exemples: two link plananr robot
    - Exemples: five link plannar robot
    - Exemples: three link robot
  - Exemples: single pendulum
  - Exemples: double pendulum
  - Exemples: cart-pole
  - Exemples: planar drone
  - Exemples: rocket
- Exemples: bicycle model (planar vehicle)


### Controller objects ###

- Linear
- PID
- LQR
- Computed-Torque
- Sliding-mode controller
- End-point impedance controller for robot arms
- End-point trajectory controller for robot arms
- Tabular look-up table controller (generated by the value-iteration algorithm)

### Planner objects ###

1. RRT tree search
2. Direct collocation trajectory optimisation
3. Dynamic programming and value-iteration 


### Analysis tool ###

- Copmuting simulation
- Phase-plane analysis
- Graphical animated output of the simulations
- Cost function computation
- Linearisation (from any dynamic class to the state-space class)
- Modal analysis
- Pole/zero computation
- Bode plot
- Reachability

<!---
## Development ##

### Installing for development ###

From the root of a clone of this repository, run:

```bash
python -m pip install -e .
```

This will install pyro in "editable" mode, so any changes made in the working
directory will be taken into account when importing the pyro module.

### Running tests ###

Note: most tests are currently broken.

Running the test suite requires installing [pytest](https://docs.pytest.org/).
To run all tests:

```bash
pytest examples/projects/tests
```

By running this command from the top level of the pyro repository, `pyro` will
be imported directly from the local folder.

--->
