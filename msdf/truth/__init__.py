import numpy as np
from numpy import sin, cos, pi


class GroundTruth:
    def __init__(self, v=300, q=9, rounds=2):
        self.v = v
        self.q = q
        self.w = q / 2*v
        self.a = v**2 / q
        self.interval = 2 / self.w * pi
        self.space = np.linspace(0, rounds * self.interval, num=rounds * 500)

    def _x(self, t):
        return sin(self.w * t)

    def _y(self, t):
        return sin(2 * self.w * t)

    def trajectory(self, t):
        return self.a * np.vstack([self._x(t), self._y(t)])

    def velocity(self, t):
        return -self.q * np.vstack([sin(self.w * t) / 4, sin(2 * self.w * t)])

    def acceleration(self, t):
        return self.v * np.vstack([cos(self.w * t) / 2, cos(2 * self.w * t)])

    def x(self, t):
        return np.vstack([self.trajectory(t), self.velocity(t), self.acceleration(t)])
