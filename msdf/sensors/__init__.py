
import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.linalg import inv
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

    def F(self, delta):
        return np.array([[1, 0, delta, 0],
                         [0, 1, 0, delta],
                         [0, 0, 1,     0],
                         [0, 0, 0,     1]])

    def D(self, delta):
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        return 500 * error

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
        return np.array([[1., 0, 0, 0],
                         [0,  1, 0, 0]])

    @property
    def R(self):
        return np.diag([self.sigma_range**2, self.sigma_azimuth**2])

    def F(self, delta):
        return np.array([[1, 0, delta, 0],
                         [0, 1, 0, delta],
                         [0, 0, 1,     0],
                         [0, 0, 0,     1]])

    def D(self, delta):
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        return 1500 * error

    @property
    def __error(self):
        return np.vstack([self.sigma_range * normal(),
                          self.sigma_azimuth * normal()])

    def velocity(self, t):
        return self.truth.velocity(t)

    def into_radar(self, vec):
        x, y = vec
        (x_sensor, y_sensor) = self.pos.flatten()
        r = sqrt((x - x_sensor) ** 2 + (y - y_sensor) ** 2)
        # arctan changes for 4 sectors of cartesian coordinates
        # +x, +y = + 0
        # -x, +y = + pi
        # -x, -y = + pi
        # +x, -y = + 2pi
        if x == x_sensor:
            return np.vstack([r, 0])
        phi = np.arctan((y - y_sensor) / (x - x_sensor))
        if x >= x_sensor:
            if y < y_sensor:
                phi += 2 * pi
        else:
            phi += pi
        return np.vstack([r, phi])

    def radar_truth(self, t):
        return self.into_radar(self.truth.trajectory(t).flatten())

    def radar(self, t):
        return self.radar_truth(t) + self.__error

    @staticmethod
    def cartesian(phi, r):
        return r * np.vstack([cos(phi), sin(phi)])

    @staticmethod
    def rotation(phi):
        return np.array([[cos(phi), -sin(phi)],
                         [sin(phi),  cos(phi)]])

    @staticmethod
    def dilation(r):
        return np.array([[1, 0],
                         [0, r]])

    def measure(self, t):
        prediction = self.radar(t)
        r, phi = prediction.flatten()
        D = self.rotation(phi)
        S = self.dilation(r)
        T = D @ S
        R = T @ self.R @ T.T
        cartesian = self.cartesian(phi, r) + T @ (self.radar_truth(t) - prediction)
        return R, cartesian + self.pos


class MergedSensor:
    def __init__(self, sensors):
        self.sensors = sensors

    @property
    def truth(self):
        return self.sensors[0].truth

    @property
    def H(self):
        return np.array([[1., 0, 0, 0],
                         [0, 1, 0, 0]])

    def R(self, matrices):
        sigma = np.zeros((2, 2))
        for m in matrices:
            sigma += inv(m)
        return inv(sigma)

    def F(self, delta):
        return np.array([[1, 0, delta, 0],
                         [0, 1, 0, delta],
                         [0, 0, 1,     0],
                         [0, 0, 0,     1]])

    def D(self, delta):
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        return error

    def measure(self, t):
        measurements = [s.measure(t) for s in self.sensors]
        R = self.R([z[0] for z in measurements])
        Z = np.zeros((2, 1))
        for m in measurements:
            Z += inv(m[0]) @ m[1]
        return R, R @ Z
