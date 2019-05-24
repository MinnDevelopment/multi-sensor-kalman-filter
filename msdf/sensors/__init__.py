
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
        return self.R, self.trajectory(t)


class RadarSensor:
    def __init__(self, pos, sigma_range=20, sigma_azimuth=0.2):
        self.truth = GroundTruth()
        self.pos = np.vstack(pos)
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

    def D(self, t, prev):
        delta = t - prev
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        return 3 * error

    @property
    def __error(self):
        return np.vstack([self.sigma_range * normal(),
                          self.sigma_azimuth * normal()])

    def velocity(self, t):
        return self.truth.velocity(t)

    def into_radar(self, x, y):
        (x_sensor, y_sensor) = self.pos.flatten()
        r = sqrt((x - x_sensor) ** 2 + (y - y_sensor) ** 2)
        # arctan changes for 4 sectors of cartesian coordinates
        # +x, +y = + 0
        # -x, +y = + pi
        # -x, -y = + pi
        # +x, -y = + 2pi
        if x == x_sensor and y == y_sensor:
            return np.vstack([r, 0]) + self.__error
        phi = np.arctan((y - y_sensor) / (x - x_sensor))
        if x >= x_sensor:
            if y < y_sensor:
                phi += 2 * pi
        else:
            phi += pi
        return np.vstack([r, phi])

    def radar(self, t):
        x, y = self.truth.trajectory(t).flatten()
        return self.into_radar(x, y) + self.__error

    def cartesian(self, phi, r):
        return r * np.vstack([cos(phi), sin(phi)]) + self.pos

    def rotation(self, phi):
        return np.array([[cos(phi), -sin(phi)],
                         [sin(phi),  cos(phi)]])

    def dilation(self, r):
        return np.array([[1, 0],
                         [0, r]])

    def measure(self, t):
        r, phi = self.radar(t).flatten()
        cartesian = self.cartesian(phi, r)
        D = self.rotation(phi)
        S = self.dilation(r)
        R = D @ S @ self.R @ S.T @ D.T
        return R, cartesian

