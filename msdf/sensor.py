import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.linalg import inv
from numpy.random.mtrand import standard_normal, normal

from msdf.truth import GroundTruth


def degree_to_radian(phi):
    return phi * (pi / 180)


angle_error = degree_to_radian(10)
range_error = 20
grid_error = 100

print("Azimuth Error:", angle_error, "rad")
print("Range Error:", range_error)
print("Grid Error:", grid_error, "\n")


class Sensor:
    __slots__ = 'truth'

    def draw(self, ax, state):
        pass

    def get_positions(self):
        raise NotImplementedError()

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
        return 4.5 * error

    def measure(self, t) -> tuple:
        raise NotImplementedError()


class GridSensor(Sensor):
    def __init__(self, sigma=grid_error):
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
    def __init__(self, pos, sigma_range=range_error, sigma_azimuth=angle_error):
        self.truth = GroundTruth()
        self.pos = np.vstack(pos)
        self.sigma_range = sigma_range
        self.sigma_azimuth = sigma_azimuth
        self.ptr = []

    def reset(self):
        for p in self.ptr:
            p.remove()
        self.ptr.clear()

    def draw(self, ax, state):
        self.reset()
        line, = ax.plot(*self.pos, "kx")
        self.ptr.append(line)

        state = state.flatten()

        x, y = self.pos.tolist()
        r, phi = self.into_radar(state)
        phi += self.sigma_azimuth
        cartesian_state = self.cartesian(r, phi) + self.pos
        x2, y2 = cartesian_state.flatten()
        x.append(x2)
        y.append(y2)
        line, = ax.plot(x, y, "b", alpha=.5)
        self.ptr.append(line)

        x, y = self.pos.tolist()
        phi -= 2 * self.sigma_azimuth
        cartesian_state = self.cartesian(r, phi) + self.pos
        x2, y2 = cartesian_state.flatten()
        x.append(x2)
        y.append(y2)
        line, = ax.plot(x, y, "b", alpha=.5)
        self.ptr.append(line)

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
        x -= x_sensor
        y -= y_sensor
        r = sqrt(x ** 2 + y ** 2)

        if x != 0:
            phi = np.arctan(y / x)
            if x < 0:
                phi += pi
            return np.vstack([r, phi])
        if y == 0:
            phi = 0
        elif y < 0:
            phi = 3*pi/2
        else:
            phi = pi/2
        return np.vstack([r, phi])

    def radar_truth(self, t):
        trajectory = self.truth.trajectory(t)
        radar = self.into_radar(trajectory.flatten())
        return radar

    def radar(self, t):
        error = self.__error
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

    def taylor(self, t):
        x = self.radar(t)
        r, phi = x.flatten()
        term = self.cartesian(r, phi)
        D = self.rotation(phi)
        S = self.dilation(r)
        T = D @ S

        def polynomial(z):
            return term + T @ (z - x)

        covariance = T @ self.R @ T.T
        return polynomial, covariance

    def measure(self, t):
        polynomial, covariance = self.taylor(t)
        truth = self.radar_truth(t)
        state = polynomial(truth) + self.pos
        return state, covariance


class MergedSensor(Sensor):
    def __init__(self, sensors):
        self.sensors = sensors

    def draw(self, ax, state):
        for s in self.sensors:
            s.draw(ax, state)

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
            sigma += inv(m)
        return inv(sigma)

    def measure(self, t):
        measurements = [s.measure(t) for s in self.sensors]
        R = self.R([z[1] for z in measurements])
        Z = np.zeros((2, 1))
        for m in measurements:
            weight = inv(m[1])
            state = m[0]
            Z += weight @ state
        return R @ Z, R