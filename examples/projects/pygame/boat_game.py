
import pygame


# pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()
pygame.joystick.init()
joysticks = []
running = True
dt = clock.tick(60) / 1000
dt = clock.tick(60) / 1000

#define font
font_size = 30
font = pygame.font.SysFont("Futura", 30)

##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import boat
from pyro.kinematic import drawing
from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.analysis import simulation
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation
##############################################################################

# Dynamic system
sys  = boat.Boat2D()

sys.mass = 1000.
sys.inertia = 1000.


x = sys.x0
t = 0.0
score = 0.0
last_score = 0.0
high_score = -100000.0

x[0] = -5
x[1] = -5

#Planner

# #Max/Min torque
# sys.u_ub[0] = +20
# sys.u_ub[1] = +20
# sys.u_lb[0] = -20
# sys.u_lb[1] = -20

sys.cost_function.Q[0,0] = 100.0
sys.cost_function.Q[0,0] = 100.0
sys.cost_function.R[0,0] = 10.0
sys.cost_function.R[0,0] = 10.0

# planner = DirectCollocationTrajectoryOptimisation( sys , 0.2 , 20 )

# planner.x_start = x
# planner.x_goal  = np.array([0,0,0,0])

# planner.maxiter = 500
# planner.set_linear_initial_guest(True)
# planner.compute_optimal_trajectory()

# # Controller
# ctl  = nonlinear.ComputedTorqueController( sys , planner.traj )
# ctl.rbar = np.array([0,0])
# ctl.w0   = 5
# ctl.zeta = 1


def px_T( domain , width = 800 , height = 800 , x_axis = 0, y_axis = 1):

    x_range = domain[x_axis][1] - domain[x_axis][0]
    y_range = domain[y_axis][1] - domain[y_axis][0]
    x_center = 0.5 * ( domain[x_axis][1] + domain[x_axis][0] )
    y_center = 0.5 * ( domain[y_axis][1] + domain[y_axis][0] )

    x_scale = width / x_range
    y_scale = height / y_range

    scale = min(x_scale,y_scale)

    x_offset = 0.5 * width - x_center * scale
    y_offset = 0.5 * height - y_center * scale

    T = np.array([ [ scale , 0 , x_offset ],
                   [  0 , -scale, y_offset],
                   [    0 , 0 , 1]])

    return T

def line2pygame( line , T ):

    line_B = drawing.transform_points_2D( T , line )

    n = line.shape[0]

    pg_pts =[]

    for i in range(n):

        pg_pts.append( (int(line_B[i,0]), int(line_B[i,1])))

    return pg_pts






while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.JOYDEVICEADDED:
            joy = pygame.joystick.Joystick(event.device_index)
            print(joy.get_name())
            joysticks.append(joy)
    
    print('joysticks:',len(joysticks))

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    u = sys.ubar
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        u[0] = sys.u_ub[0] * 1.0
    elif keys[pygame.K_d]:
        u[0] = sys.u_lb[0] * 1.0
    else:
        u[0] = 0.0
        # vert_move = joystick.get_axis(1)

    # reset
    if t > 10.01:
        last_score = score
        high_score = max(high_score,last_score)
        score = 0.0
        x = sys.x0
        # x = np.random.normal( sys.x0, np.array([0.5,1.0,0.5,0.5]))
        x[0] = -np.pi
        # x = x + np.
        t = 0.0
    
    if keys[pygame.K_w]:
        u[1] = sys.u_ub[1] * 1.0
    elif keys[pygame.K_s]:
        u[1] = sys.u_lb[1] * 1.0
    else:
        u[1] = 0.0

    for joy in joysticks:
        u[0] = joy.get_axis(1) * ( sys.u_ub[0] - sys.u_lb[0] ) * 0.5
        u[1] = joy.get_axis(3) * ( sys.u_ub[1] - sys.u_lb[1] ) * 0.5

        # if joy.get_button(0):
        #     u = ctl.c( x , ctl.rbar, t )

    # if keys[pygame.K_o]:
    #     u = ctl.c( x , ctl.rbar, t )

    # Dynamic
    x = sys.x_next( x , u , t , dt )
    score = score - sys.cost_function.g( x , u , t ) * dt / 100
    t = t + dt

    # Kinematic
    q = sys.xut2q( x , u , 0 )

    print('x:',x,'u:,',u, 'dt:', dt, 't:',t)
    print('score: %.2f' % score)

    img = font.render( 'Score: %.2f' % score, True, (0, 0, 0))
    screen.blit(img, (0,0))

    img = font.render( 'Last score: %.2f' % last_score, True, (0, 0, 0))
    screen.blit(img, (0,40))

    img = font.render( 'High score: %.2f' % high_score, True, (0, 0, 0))
    screen.blit(img, (0,80))

    img = font.render( 't= %.2f' % t, True, (0, 0, 0))
    screen.blit(img, (0,120))

    # Graphical
    lines, _ , _ = sys.forward_kinematic_lines( q )

    domain = sys.forward_kinematic_domain( q )

    body = lines[0]


    T = px_T( domain )

    body_px = line2pygame( body , T )
    pygame.draw.lines( screen, 'blue',False, body_px , width = 5  )

    force = sys.forward_kinematic_lines_plus(x,u,0)[0][0]
    if force.shape[0] > 2:
        force_px = line2pygame( force , T )
        pygame.draw.aalines( screen, 'red',False, force_px ) #, width = 3 )

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()