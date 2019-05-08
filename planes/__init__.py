from numpy import sin, cos, pi, sqrt
from numpy.random import standard_normal
from time import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class GroundTruth:
    def __init__(self, v, q, rounds=1):
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

    def acceleration(self, t):
        return self.v * np.array([[ cos(self.w * t) / 2 ],
                                  [ cos(2 * self.w * t) ]])

    def velocity(self, t):
        return -self.q * np.array([[ sin(self.w * t) / 4 ],
                                   [ sin(2 * self.w * t) ]])


class Sensor:
    def __init__(self, error=50, interval=5):
        self.truth = GroundTruth(300, 9)
        self.sigma = error
        self.interval = interval
        self.real_point = None

        self._next_read = 0
        self._point = None
        self._accel = None
        self._veloc = None
        self._ax = None
        self._circle = None
        self._radius = self.sigma * 3

    def get_error(self):
        vec = standard_normal((2, 1))
        print("Normal: ", vec.flatten())
        return self.sigma * vec

    def get_trajectory_data(self, t):
        return self.truth.trajectory(t) + self.get_error()

    def get_acceleration_data(self, t):
        return self.truth.acceleration(t) # + self.sigma * normal(size=(2, 1))

    def get_velocity_data(self, t):
        return self.truth.velocity(t) # + self.sigma * normal(size=(2, 1))

    def plot_frame(self, t):
        real_position = self.truth.trajectory(t)
        self.real_point.set_data(real_position)
        if self._next_read > time():
            return self._point
        else:
            return self.plot_trajectory(t, real_position)

    def plot_init(self, ax):
        self._ax = ax
        self._point, = ax.plot([0], [0], "ro")
        self.real_point, = ax.plot([0], [0], "go")
        # plot track function
        data = [self.truth.trajectory(t) for t in self.truth.space]
        t, d = ([p[0] for p in data], [p[1] for p in data])
        ax.plot(t, d, 'grey', alpha=0.5)

        self._accel = ax.arrow(0, 0, 0, 0, width=100, color="r")
        self._veloc = ax.arrow(0, 0, 0, 0, width=100, color="b")

    def plot_trajectory(self, t, real_pos):
        self._next_read = time() + self.interval
        error = self.get_error()
        pos = error + real_pos
        print("Error: ", error.flatten())
        self._point.set_data(pos)
        if self._circle:
            self._circle.remove()
        self._circle = plt.Circle(pos, self._radius, color="r", fill=False)
        self._ax.add_artist(self._circle)
        self.plot_arrows(t, pos)
        return self._point

    def plot_arrows(self, t, pos):
        a = (10 * self.get_acceleration_data(t)).flatten().tolist()
        b = (200 * self.get_velocity_data(t)).flatten().tolist()
        self._accel.remove()
        self._veloc.remove()
        (x, y) = pos.flatten()
        self._accel = self._ax.arrow(x, y, *a, width=100, color="r")
        self._veloc = self._ax.arrow(x, y, *b, width=100, color="b")


class KalmanFilter:
    def __init__(self, sensor, sigma):
        self.sensor = sensor
        self.sigma = sigma
        self._r = self.sigma**2 * np.eye(2)


def main():
    sensor = Sensor()
    frames = sensor.truth.space

    fig, ax = plt.subplots()
    ax.set_ylim(-11000, 11000)
    ax.set_xlim(-11000, 11000)
    sensor.plot_init(ax)
    ani = animation.FuncAnimation(fig, sensor.plot_frame, frames, repeat=True, interval=10)
    plt.show()


if __name__ == '__main__':
    main()
