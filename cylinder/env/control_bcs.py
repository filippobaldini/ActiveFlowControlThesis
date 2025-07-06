from math import sqrt, pi, atan2, sin, cos
from dolfin import *
import numpy as np


class JetBCValue(UserExpression):
    def __init__(self, time, center, Q, theta0, width, radius=0.5, **kwargs):
        super().__init__(**kwargs)
        self.time = time
        self.center = center
        self.Q = Q
        self.theta0 = theta0
        self.frequency = 1.0  # frequency of oscillation
        self.width = width
        self.radius = radius  # fixed cylinder radius

    def eval(self, values, x):
        dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        r = self.radius
        norm = sqrt(dx**2 + dy**2)
        if norm < 1e-10:
            values[0], values[1] = 0.0, 0.0
            return
        nx, ny = dx / norm, dy / norm  # radial direction

        # Angular position
        theta = atan2(dy, dx)
        delta_theta = theta - self.theta0
        delta_theta = atan2(sin(delta_theta), cos(delta_theta))  # keep in [-π, π]

        if abs(delta_theta) > self.width / 2:
            amp = 0.0
        else:
            s = delta_theta * r
            half_arc = (self.width * r) / 2
            amp = self.Q * (1 - (s / half_arc) ** 2)
            amp *= cos(pi * delta_theta / self.width)  # sin(2 * pi * 0.6 *self.time)

        values[0] = amp * nx
        values[1] = amp * ny

    def value_shape(self):
        return (2,)


# def normalize_angle(angle):
#     '''Make angle in [-pi, pi]'''
#     assert angle >= 0

#     if angle < pi:
#         return angle
#     if angle < 2*pi:
#         return -((2*pi)-angle)

#     return normalize_angle(angle - 2*pi)

# class JetBCValue(Expression):
#     '''
#     Value of this expression is a vector field v(x, y) = A(theta)*e_r
#     where A is the amplitude function of the polar angle and e_r is radial
#     unit vector. The field is modulated such that

#     1) at theta = theta0 \pm width/2 A is 0
#     2) \int_{J} v.n dl = Q

#     Here theta0 is the (angular) position of the jet on the cylinder, width
#     is its angular width and finaly Q is the desired flux thought the jet.
#     All angles are in degrees.
#     '''
#     def __init__(self, radius, width, theta0, Q, **kwargs):
#         assert width > 0 and radius > 0 # Sanity. Allow negative Q for suction
#         # theta0 = np.deg2rad(theta0)
#         assert theta0 >= 0  # As coming from deg to rad

#         self.radius = radius
#         # self.width = np.deg2rad(width)
#         # From deg2rad it is possible that theta0 > pi. Below we habe atan2 so
#         # shift to -pi, pi
#         self.theta0 = normalize_angle(theta0)

#         self.Q = Q

#     def eval(self, values, x):
#         A = self.amplitude(x)
#         xC = 0.
#         yC = 0.

#         values[0] = A*(x[0] - xC)
#         values[1] = A*(x[1] - yC)

#     def amplitude(self, x):
#         theta = np.arctan2(x[1], x[0])

#         # NOTE: motivation for below is cos(pi*(theta0 \pm width)/w) = 0 to
#         # smoothly join the no slip.
#         scale = self.Q/(2.*self.width*self.radius**2/pi)

#         return scale*cos(pi*(theta - self.theta0)/self.width)

#     # This is a vector field in 2d
#     def value_shape(self):
#         return (2, )
