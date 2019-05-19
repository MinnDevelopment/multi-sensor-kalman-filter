from threading import Timer

import numpy as np
from numpy import sin, cos, pi, sqrt
from numpy.random import standard_normal, normal


class GroundTruth:
    def __init__(self, v=300, q=9, rounds=1):
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
        return self.a * np.array([[ self._x(t) ],
                                  [ self._y(t) ]])

    def velocity(self, t):
        return -self.q * np.array([[ sin(self.w * t) / 4 ],
                                   [ sin(2 * self.w * t) ]])

    def acceleration(self, t):
        return self.v * np.array([[ cos(self.w * t) / 2 ],
                                  [ cos(2 * self.w * t) ]])

    def x(self, t):
        return np.vstack([self.trajectory(t), self.velocity(t), self.acceleration(t)])


class GenericSensor:
    def __init__(self, sigma=50, interval=5):
        self.truth = GroundTruth(300, 9)
        self.sigma = sigma
        self.interval = interval
        self.real_point = None
        self._traj_timer = None

    def __schedule(self, fn):
        timer = Timer(self.interval, fn)
        timer.start()
        return timer

    def reset(self):
        if self._traj_timer:
            self._traj_timer.cancel()

    def on_trajectory(self, callback, at):
        def runner():
            callback(self.get_trajectory_data(at()))
        if self._traj_timer:
            self._traj_timer.cancel()
        self._traj_timer = self.__schedule(runner)

    def get_error(self):
        return self.sigma * standard_normal((2, 1))

    def get_trajectory_data(self, t):
        return self.truth.trajectory(t) + self.get_error()


class PositionalSensor:
    def __init__(self, pos, sigma_range=20, sigma_azimuth=0.2):
        self.truth = GroundTruth()
        self.pos = pos
        self.sigma_range = sigma_range
        self.sigma_azimuth = sigma_azimuth

    def get_error(self):
        return np.array([self.sigma_range * normal(),
                         self.sigma_azimuth * normal()]).T

    def get_range_azimuth(self, t):
        (x_real, y_real) = self.truth.trajectory(t).flatten()
        (x_sensor, y_sensor) = self.pos
        new_x = sqrt((x_real - x_sensor)**2 + (y_real - y_sensor)**2)
        new_y = np.arctan((y_real - y_sensor) / (x_real - x_sensor))
        return np.array([new_x, new_y]).T



def main():
    print(PositionalSensor((0, 0)).get_range_azimuth(100))


if __name__ == '__main__':
    main()
