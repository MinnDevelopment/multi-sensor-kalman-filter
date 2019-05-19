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
        return self.a * np.vstack([self._x(t), self._y(t)])

    def velocity(self, t):
        return -self.q * np.array([[ sin(self.w * t) / 4 ],
                                   [ sin(2 * self.w * t) ]])

    def acceleration(self, t):
        return self.v * np.array([[ cos(self.w * t) / 2 ],
                                  [ cos(2 * self.w * t) ]])

    def x(self, t):
        return np.vstack([self.trajectory(t), self.velocity(t), self.acceleration(t)])


class PositionalSensor:
    def __init__(self, pos, sigma=50, sigma_range=20, sigma_azimuth=0.2):
        self.truth = GroundTruth()
        self.pos = pos
        self.sigma = sigma
        self.sigma_range = sigma_range
        self.sigma_azimuth = sigma_azimuth

    @property
    def H(self):
        return np.array([[1., 0], [0, 1], [0, 0], [0, 0]]).T

    def get_range_error(self):
        return np.vstack([self.sigma_range * normal(),
                          self.sigma_azimuth * normal()])

    def get_trajectory_error(self):
        return self.sigma * standard_normal((2, 1))

    def get_velocity(self, t):
        return self.truth.velocity(t)

    def get_trajectory_data(self, t):
        error = self.get_trajectory_error()
        trajectory = self.truth.trajectory(t)
        return trajectory + error

    def get_range_azimuth_data(self, t):
        (x_real, y_real) = self.truth.trajectory(t).flatten()
        (x_sensor, y_sensor) = self.pos
        new_x = sqrt((x_real - x_sensor)**2 + (y_real - y_sensor)**2)
        new_y = np.arctan((y_real - y_sensor) / (x_real - x_sensor))
        return np.array([new_x, new_y]).T

    def measure(self, t):
        return np.vstack((self.get_trajectory_data(t), self.get_velocity(t)))


class KalmanFilter:
    def __init__(self, sensors, matrix=50**2 * np.diag((1, 1))):
        self.sensors = sensors
        self.matrix = matrix

    def prediction(self, t):
        pass

    def filtering(self, measurement):
        pass


def main():
    sensor = PositionalSensor((0, 0))
    filter = KalmanFilter([sensor])
    for t in sensor.truth.space:
        measure = sensor.measure(t)
        print(t, "\n", measure)
        print(sensor.H @ measure)
        print("----")


if __name__ == '__main__':
    main()
