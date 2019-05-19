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
        return -self.q * np.vstack([sin(self.w * t) / 4, sin(2 * self.w * t)])

    def acceleration(self, t):
        return self.v * np.vstack([cos(self.w * t) / 2, cos(2 * self.w * t)])

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
    def __init__(self):
        self.prev_t = 0
        self.prev_estimate = np.zeros((4, 1))
        self.prev_ignorance = np.eye(4)

    def F(self, k):
        delta = k - self.prev_t
        return np.array([[1, 0, delta, 0],
                         [0, 1, 0, delta],
                         [0, 0, 1,     0],
                         [0, 0, 0,     1]])

    def D(self, k):
        delta = k - self.prev_t
        delta4 = delta ** 4 / 4
        delta3 = delta ** 3 / 2
        delta2 = delta ** 2
        error = np.array([[delta4, 0, delta3, 0],
                          [0, delta4, 0, delta3],
                          [delta3, 0, delta2, 0],
                          [0, delta3, 0, delta2]])
        return error

    def prediction(self, t):
        transition = self.F(t)
        estimate = transition @ self.prev_estimate
        ignorance = transition @ self.prev_ignorance @ transition.T + self.D(t)
        return estimate, ignorance

    def filtering(self, t, measurement, H, R):
        estimate, ignorance = self.prediction(t)
        v = measurement - H @ estimate
        gain = H @ ignorance @ H.T + R

        W = ignorance @ H.T @ np.linalg.inv(gain)

        self.prev_estimate = estimate + W @ v
        self.prev_ignorance = ignorance - W @ gain @ W.T


def main():
    sensor = PositionalSensor((0, 0))
    filter = KalmanFilter()
    count = 0
    R = 2500 * np.diag([1, 1])
    for t in sensor.truth.space:
        count += 1
        if count % 5 == 0:
            print("Filtering...")
            filter.filtering(t, sensor.get_trajectory_data(t), sensor.H, R)
        print("Actual:\n")
        print(sensor.truth.trajectory(t))
        print("Prediction:\n")
        prediction, _ = filter.prediction(t)
        print(prediction)
        print("============\n")


if __name__ == '__main__':
    main()
