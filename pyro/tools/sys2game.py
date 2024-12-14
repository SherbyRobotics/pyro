# -*- coding: utf-8 -*-

##############################################################################
import pygame
import numpy as np
from pyro.kinematic import drawing

##############################################################################


##############################################w#################################
class InteractiveContinuousDynamicSystem:

    ############################
    def __init__(self, sys, tf=10.0, dt=0.01, ctl=None, renderer="pygame"):

        self.sys = sys
        self.dt = dt
        self.tf = tf

        self.ctl = ctl
        self.renderer = renderer

        # Parameters
        self.input_scale = 1.0
        self.input_axis_mapping = [0, 2, 1, 3, 4, 5]
        self.dot_size = 20
        self.linewidth = 10

        self.reset_memory()
        self.high_score = -np.inf
        self.reset_score()
        self.init_event()
        self.init_renderer()

    ############################
    def reset_memory(self, t=0.0):

        self.x = self.sys.x0
        self.t = t
        self.u = self.sys.ubar
        self.y = self.sys.h(self.x, self.u, self.t)

    ############################
    def reset_score(self, last_score=-np.inf):

        self.score = 0.0
        self.last_score = last_score
        self.high_score = max(self.high_score, last_score)

    ############################
    def init_event(self):

        pygame.init()
        pygame.joystick.init()
        self.joysticks = []
        self.clock = pygame.time.Clock()
        self.running = True
        self.stop = False

        self.dt = self.clock.tick(60) / 1000

    ############################
    def init_renderer(self, width=800, height=800):

        if self.renderer == "pygame":

            self.screen = pygame.display.set_mode((width, height))
            self.font = pygame.font.SysFont("Futura", 30)
            self.width = width
            self.height = height

        else:
            raise ValueError("Renderer not supported.")

    ############################
    def run(self, debug=False):

        while self.running:

            self.process_events()
            self.step()
            self.render()

            if self.t > self.tf or self.stop:
                self.reset_memory()
                self.reset_score(self.score)

            if debug:
                print("x:", self.x, "u:,", self.u, "dt:", self.dt, "t:", self.t)
                print("score: %.2f" % self.score)

        pygame.quit()

    ############################
    def process_events(self):

        self.dt = self.clock.tick(60) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.JOYDEVICEADDED:
                joy = pygame.joystick.Joystick(event.device_index)
                print(joy.get_name())
                self.joysticks.append(joy)

        # Keys inputs
        u = self.sys.ubar

        keys = pygame.key.get_pressed()

        for j in range(self.sys.ubar.shape[0]):

            if j == 0:
                if keys[pygame.K_a]:
                    u[0] = self.sys.u_ub[0] * self.input_scale
                elif keys[pygame.K_d]:
                    u[0] = self.sys.u_lb[0] * self.input_scale
                else:
                    u[0] = 0.0

            if j == 1:
                if keys[pygame.K_w]:
                    u[1] = self.sys.u_ub[1] * self.input_scale
                elif keys[pygame.K_s]:
                    u[1] = self.sys.u_lb[1] * self.input_scale
                else:
                    u[1] = 0.0

            if j == 2:
                if keys[pygame.K_o]:
                    u[2] = self.sys.u_ub[2] * self.input_scale
                elif keys[pygame.K_l]:
                    u[2] = self.sys.u_lb[2] * self.input_scale
                else:
                    u[2] = 0.0

            if j == 3:
                if keys[pygame.K_i]:
                    u[3] = self.sys.u_ub[3] * self.input_scale
                elif keys[pygame.K_k]:
                    u[3] = self.sys.u_lb[3] * self.input_scale
                else:
                    u[3] = 0.0

        # Joystick inputs
        for joy in self.joysticks:

            # Stop button
            self.stop = joy.get_button(3)

            for j in self.sys.ubar.shape[0]:
                input = self.input_axis_mapping[j]
                u[j] = input * (sys.u_ub[j] - self.sys.u_lb[j]) * self.input_scale

        # Automatic modes
        if self.ctl is not None:

            t = self.t
            y = self.y
            r = self.ctl.rbar

            # Joystick inputs
            for joy in self.joysticks:
                if joy.get_button(0):
                    u = self.ctl.c(y, r, t)
                    u = np.clip(u, self.sys.u_lb, self.sys.u_ub)

                if joy.get_button(1):
                    u = self.ctl.c(y, r, t)

            if keys[pygame.K_o]:
                u = self.ctl.c(y, r, t)

        # Update sys inputs
        self.u = u

    ############################
    def step(self):

        # Score update
        self.score = (
            self.score - self.sys.cost_function.g(self.x, self.u, self.t) * self.dt
        )

        # Dynamic
        x_next = self.sys.x_next(self.x, self.u, self.t, self.dt)
        self.t = self.t + self.dt
        self.x = x_next

        # Output
        self.y = self.sys.h(self.x, self.u, self.t)

    ############################
    def render(self):

        if self.renderer == "pygame":

            self.screen.fill("white")

            # Score
            img = self.font.render("Score: %.2f" % self.score, True, (0, 0, 0))
            self.screen.blit(img, (0, 0))

            img = self.font.render(
                "Last score: %.2f" % self.last_score, True, (0, 0, 0)
            )
            self.screen.blit(img, (0, 40))

            img = self.font.render(
                "High score: %.2f" % self.high_score, True, (0, 0, 0)
            )
            self.screen.blit(img, (0, 80))

            img = self.font.render("t= %.2f" % self.t, True, (0, 0, 0))
            self.screen.blit(img, (0, 120))

            # Kinematic
            q = self.sys.xut2q(self.x, self.u, self.t)
            lines, _, _ = self.sys.forward_kinematic_lines(q)
            domain = self.sys.forward_kinematic_domain(q)

            # pixel transformation
            T = self.px_T(domain)

            for line in lines:
                if line.shape[0] > 1:
                    line_px = self.line2pygame(line, T)
                    pygame.draw.lines(
                        self.screen, "blue", False, line_px, width=self.linewidth
                    )

                    if self.dot_size > 0:
                        for i in range(line.shape[0]):
                            xyz = np.array([line[i, 0], line[i, 1], 1.0])
                            px = T @ xyz
                            pygame.draw.circle(
                                self.screen, "blue", px[0:2], self.dot_size
                            )

            # Lines plus
            forces = self.sys.forward_kinematic_lines_plus(self.x, self.u, self.t)[0]

            for force in forces:
                if force.shape[0] > 2:
                    force_px = self.line2pygame(force, T)
                    pygame.draw.aalines(
                        self.screen, "red", False, force_px  # , width=self.linewidth
                    )

            pygame.display.flip()

    ################################
    def px_T(self, domain, x_axis=0, y_axis=1):

        width = self.width
        height = self.height

        x_range = domain[x_axis][1] - domain[x_axis][0]
        y_range = domain[y_axis][1] - domain[y_axis][0]
        x_center = 0.5 * (domain[x_axis][1] + domain[x_axis][0])
        y_center = 0.5 * (domain[y_axis][1] + domain[y_axis][0])

        x_scale = width / x_range
        y_scale = height / y_range

        scale = min(x_scale, y_scale)

        x_offset = 0.5 * width - x_center * scale
        y_offset = 0.5 * height + y_center * scale

        T = np.array([[scale, 0, x_offset], [0, -scale, y_offset], [0, 0, 1]])

        return T

    ################################
    def line2pygame(self, line, T):

        line_B = drawing.transform_points_2D(T, line)

        n = line.shape[0]

        pg_pts = []

        for i in range(n):

            pg_pts.append((int(line_B[i, 0]), int(line_B[i, 1])))

        return pg_pts


if __name__ == "__main__":
    """MAIN TEST"""

    from pyro.dynamic import pendulum
    from pyro.dynamic import drone
    from pyro.dynamic import boat
    from pyro.dynamic import vehicle_steering
    from pyro.dynamic import rocket
    from pyro.dynamic import mountaincar
    from pyro.dynamic import massspringdamper
    from pyro.dynamic import plane

    sys = pendulum.DoublePendulum()
    sys = drone.Drone2D()
    sys = pendulum.SinglePendulum()
    sys = boat.Boat2D()
    # sys = vehicle_steering.KinematicCarModelwithObstacles() # bug?
    sys = rocket.Rocket()
    sys = mountaincar.MountainCar()
    sys = massspringdamper.ThreeMass()
    sys = plane.Plane2D()

    # sys.x0[0] = -np.pi

    sys.inertia = 10.0

    game = InteractiveContinuousDynamicSystem(sys)

    game.dot_size = 1.0

    game.run(debug=True)
