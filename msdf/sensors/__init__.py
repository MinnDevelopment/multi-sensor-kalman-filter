
import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.linalg import pinv
from numpy.random import normal, standard_normal

from msdf.truth import GroundTruth


class Sensor:
    __slots__ = 'truth'

    def get_positions(self):
        raise NotImplementedError

    @property
    def H(self):
        return np.array([[1., 0, 0, 0],
                         [0,  1, 0, 0]])

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
        # Choose sigma from [4.5, 9]
        return 6.75 * error

    def measure(self, t):
        raise NotImplementedError


class GridSensor(Sensor):
    def __init__(self, sigma=50):
        self.sigma = sigma

    def get_positions(self):
        return None

    @property
    def R(self):
        return self.sigma**2 * np.diag([1, 1])

    @property
    def __error(self):
        return self.sigma * standard_normal((2, 1))

    def trajectory(self, t):
        return self.truth.trajectory(t) + self.__error

    def measure(self, t):
        return self.trajectory(t), self.R


class RadarSensor(Sensor):
    def __init__(self, pos, sigma_range=20, sigma_azimuth=0.2):
        self.truth = GroundTruth()
        self.pos = np.vstack(pos)
        self.sigma_range = sigma_range
        self.sigma_azimuth = sigma_azimuth

    def get_positions(self):
        return [self.pos]

    @property
    def R(self):
        return np.diag([self.sigma_range**2, self.sigma_azimuth**2])

    @property
    def __error(self):
        return np.vstack([self.sigma_range * normal(),
                          self.sigma_azimuth * normal()])

    def into_radar(self, vec):
        x, y = vec
        (x_sensor, y_sensor) = self.pos.flatten()
        r = sqrt((x - x_sensor) ** 2 + (y - y_sensor) ** 2)
        # arctan changes for 4 sectors of cartesian coordinates
        # +x, +y = + 0
        # -x, +y = + pi
        # -x, -y = + pi
        # +x, -y = + 2pi
        if x == x_sensor and y == y_sensor:
            return np.vstack([r, 0])
        phi = np.arctan((y - y_sensor) / (x - x_sensor))
        if x >= x_sensor:
            if y < y_sensor:
                phi += 2 * pi
        else:
            phi += pi
        return np.vstack([r, phi])

    def radar_truth(self, t):
        trajectory = self.truth.trajectory(t)
        radar = self.into_radar(trajectory.flatten())
        return radar

    def radar(self, t):
        error = self.__error
        print("Radar error", error.flatten())
        return self.radar_truth(t) + error

    @staticmethod
    def cartesian(r, phi):
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
        truth = self.radar_truth(t)
        r, phi = truth.flatten()
        D = self.rotation(phi)
        S = self.dilation(r)
        T = D @ S
        R = T @ self.R @ T.T
        cartesian = self.cartesian(r, phi) + T @ (prediction - truth)
        return cartesian + self.pos, R


class MergedSensor(Sensor):
    def __init__(self, sensors):
        self.sensors = sensors

    def get_positions(self):
        positions = []
        for s in self.sensors:
            for p in s.get_positions() or []:
                if p is not None:
                    positions.append(p)
        return positions

    @property
    def truth(self):
        return self.sensors[0].truth

    def R(self, matrices):
        sigma = np.zeros((2, 2))
        for m in matrices:
            sigma += pinv(m)
        return pinv(sigma)

    def measure(self, t):
        measurements = [s.measure(t) for s in self.sensors]
        R = self.R([z[1] for z in measurements])
        Z = np.zeros((2, 1))
        for m in measurements:
            Z += pinv(m[1]) @ m[0]
        return R @ Z, R
