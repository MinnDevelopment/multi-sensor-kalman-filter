
import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.random import normal, standard_normal

from msdf.truth import GroundTruth


class GridSensor:
    def __init__(self, sigma=50):
        self.truth = GroundTruth()
        self.sigma = sigma

    @property
    def H(self):
        return np.array([[1., 0, 0, 0],
                         [0,  1, 0, 0]])

    @property
    def R(self):
        return self.sigma**2 * np.diag([1, 1])

    def F(self, t, prev):
        delta = t - prev
        return np.array([[1, 0, delta, 0],
                         [0, 1, 0, delta],
                         [0, 0, 1,     0],
                         [0, 0, 0,     1]])

    def D(self, t, prev):
        delta = t - prev
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        return error

    @property
    def __error(self):
        return self.sigma * standard_normal((2, 1))

    def trajectory(self, t):
        return self.truth.trajectory(t) + self.__error

    def velocity(self, t):
        return self.truth.velocity(t)

    def measure(self, t):
        return np.vstack([self.trajectory(t), self.velocity(t)])


class RadarSensor:
    def __init__(self, pos, sigma_range=20, sigma_azimuth=0.2):
        self.truth = GroundTruth()
        self.pos = pos
        self.sigma_range = sigma_range
        self.sigma_azimuth = sigma_azimuth

    @property
    def H(self):
        return np.array([[1., 0], [0, 1], [0, 0], [0, 0]]).T

    @property
    def R(self):
        return np.diag([self.sigma_range**2, self.sigma_azimuth**2])

    def F(self, t, prev):
        delta = t - prev
        return np.array([[1, 0, delta, 0],
                         [0, 1, 0, delta],
                         [0, 0, 1,     0],
                         [0, 0, 0,     1]])

    # TODO: Change this
    def D(self, t, prev):
        delta = t - prev
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        return error

    @property
    def __error(self):
        return np.vstack([self.sigma_range * normal(),
                          self.sigma_azimuth * normal()])

    def velocity(self, t):
        return self.truth.velocity(t)

    def radar(self, t):
        (x_real, y_real) = self.truth.trajectory(t).flatten()
        (x_sensor, y_sensor) = self.pos
        new_x = sqrt((x_real - x_sensor)**2 + (y_real - y_sensor)**2)
        # arctan changes for 4 sectors of cartesian coordinates
        # +x, +y = + 0
        # -x, +y = + pi
        # -x, -y = + pi
        # +x, -y = + 2pi
        if x_real == x_sensor and y_real == y_sensor:
            return np.vstack([new_x, 0]) + self.__error
        new_y = np.arctan((y_real - y_sensor) / (x_real - x_sensor))
        if x_real >= x_sensor:
            if y_real < y_sensor:
                new_y += 2 * pi
        else:
            new_y += pi
        return np.vstack([new_x, new_y]) + self.__error

    def measure_cartesian(self, t):
        r, a, _, _ = self.measure(t)
        return r * np.vstack([cos(a), sin(a)])

    def measure(self, t):
        return np.vstack([self.radar(t), self.velocity(t)])

