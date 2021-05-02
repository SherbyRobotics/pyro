# Python Robotics

A object-based toolbox for robot dynamic simulation, analysis, control and planning. 

## Installation ##

### Dependencies ####
* numpy
* scipy
* matplotlib

### Recommended environment ###
Anaconda distribution + spyder IDE available here: https://www.anaconda.com/products/individual

Note: If graphical animations are not working, try changing the graphics backend. In spyder this option is found in the menu at python/Preferences/IPython console/Backend. Inline does not allow animations, it is best to use Automatic of OS X (for Mac).

### Clone repo and add to python path ###

A simple option for development is simply to clone the repo:
```bash
git clone https://github.com/SherbyRobotics/pyro.git
```
then add the pyro folder to the pythonpath variable of your environment. In spyder this option is found in the menu at python/PYTHONPATH manager.

## Library Architecture ##

### Dynamic objects ###

At the core of pyro is a mother-class representing generic non-linear dynamic systems, with the following nomemclature:

<img width="929" alt="Screen Shot 2021-05-02 at 15 57 47" src="https://user-images.githubusercontent.com/16725496/116826021-fd9b7a80-ab5f-11eb-8e50-d7361094cbee.png">

Other more specific mother-class are 
-Linear System
-Mechanical System
-Manipulator Robot

<img width="763" alt="Screen Shot 2021-05-02 at 16 13 51" src="https://user-images.githubusercontent.com/16725496/116826418-dc3b8e00-ab61-11eb-9372-09ae08f0b15a.png">


## Controller objects ###

Controller objects can be used to closed the loop with an operation generating a closed-loop dynamic system:

closed-loop system = controller + open-loop system

For "memoryless" controller, this operation is

<img width="760" alt="Screen Shot 2021-05-02 at 16 17 34" src="https://user-images.githubusercontent.com/16725496/116826519-59ff9980-ab62-11eb-8256-6a9f4a3f4f0f.png">


## How to use ##

See exemples scripts in pyro/exemples.

Coming soon..










