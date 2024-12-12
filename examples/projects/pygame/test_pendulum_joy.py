
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

##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import pendulum
from pyro.kinematic import drawing
##############################################################################

# Dynamic system
sys  = pendulum.SinglePendulum()

sys.lc1 = 5.0
sys.u_ub[0] = +30.0
sys.u_lb[0] = -30.0
x = sys.x0

scale = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)


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
        u[0] = sys.u_ub[0] * 0.5
    elif keys[pygame.K_d]:
        u[0] = sys.u_lb[0] * 0.5
    else:
        u[0] = 0.0
        # vert_move = joystick.get_axis(1)

    for joy in joysticks:
        u[0] = joy.get_axis(0) * ( sys.u_ub[0] - sys.u_lb[0] ) * 0.5

    # Dynamic
    x = sys.x_next( x , u , 0 , dt )

    # Kinematic
    q = sys.xut2q( x , u , 0 )

    print('q:',q,'u:,',u, 'dt:',dt)

    # Graphical
    lines, _ , _ = sys.forward_kinematic_lines( q )

    domain = sys.forward_kinematic_domain( q )

    body = lines[1]


    T = px_T( domain )

    xyz = np.array([ body[1,0] , body[1,1] , 1.0 ])
    px = T @ xyz
    pygame.draw.circle(screen, 'blue', px[0:2], 20)

    body_px = line2pygame( body , T )
    pygame.draw.lines( screen, 'blue',True, body_px , width = 5  )

    force = sys.forward_kinematic_lines_plus(x,u,0)[0][0]
    if force.shape[0] > 2:
        force_px = line2pygame( force , T )
        pygame.draw.aalines( screen, 'red',False, force_px ) #, width = 3 )

    force = sys.forward_kinematic_lines_plus(x,u,0)[0][1]
    force_px = line2pygame( force , T )
    pygame.draw.aalines( screen, 'red',False, force_px ) #, width = 3  )


    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()