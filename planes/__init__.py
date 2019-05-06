
from time import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi
from numpy.random import normal


class GroundTruth:
    def __init__(self, v, q, rounds=1):
        self.v = v
        self.q = q
        self.w = q / 2*v
        self.a = v**2 / q
        self.interval = 2 / self.w * pi
        self.space = np.linspace(0, rounds * self.interval, num=rounds * 500)
        self.ptr = []

    def plot(self):
        fig, ax = plt.subplots()
        # create object
        point, = ax.plot([0], [0], 'go')

        def init():
            # setup plane
            ax.set_ylim(-11000, 11000)
            ax.set_xlim(-11000, 11000)
            # plot track function
            data = [self.trajectory(t) for t in self.space]
            t, d = ([p[0] for p in data], [p[1] for p in data])
            ax.plot(t, d, 'grey', alpha=0.5)
            # plot initial position
            point.set_data(0, 0)
            return point

        def frame(data):
            # t = time
            # x = x position
            # y = y position
            t, x, y = data

            # get current trajectory and scale it for visibility
            a = (10 * self.acceleration(t)).flatten().tolist()
            b = (200 * self.velocity(t)).flatten().tolist()

            # clear previous frame
            for p in self.ptr:
                p.remove()
            self.ptr.clear()
            # plot acceleration and velocity as arrows
            self.ptr.append(ax.arrow(*x, *y, *a, width=100, color="r"))
            self.ptr.append(ax.arrow(*x, *y, *b, width=100, color="b"))
            # plot new position
            point.set_data(x, y)
            return point

        movement = ((t, *self.trajectory(t)) for t in self.space)
        ani = animation.FuncAnimation(fig, frame, movement, init_func=init, interval=10)
        plt.show()

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
        self._next_read = 0
        self._point = None
        self._accel = None
        self._veloc = None
        self._ax = None
        self.real_point = None

    def init(self, ax):
        self._ax = ax
        self._point, = ax.plot([0], [0], "ro")
        self.real_point, = ax.plot([0], [0], "go")
        # plot track function
        data = [self.truth.trajectory(t) for t in self.truth.space]
        t, d = ([p[0] for p in data], [p[1] for p in data])
        ax.plot(t, d, 'grey', alpha=0.5)

        self._accel = ax.arrow(0, 0, 0, 0, width=100, color="r")
        self._veloc = ax.arrow(0, 0, 0, 0, width=100, color="b")

    def get_position_data(self, t):
        return self.truth.trajectory(t) + self.sigma * normal()

    def draw(self, t):
        real_position = self.truth.trajectory(t)
        self.real_point.set_data(real_position)
        if self._next_read > time():
            return self._point
        else:
            return self.plot_position(t)

    def plot_position(self, t):
        self._next_read = time() + self.interval
        pos = self.get_position_data(t)
        self._point.set_data(pos)
        self.plot_arrows(t, pos)
        return self._point

    def plot_arrows(self, t, pos):
        # TODO: RV change for velocity/acceleration vectors
        a = (10 * self.truth.acceleration(t)).flatten().tolist()
        b = (200 * self.truth.velocity(t)).flatten().tolist()
        self._accel.remove()
        self._veloc.remove()
        (x, y) = pos.flatten()
        self._accel = self._ax.arrow(x, y, *a, width=100, color="r")
        self._veloc = self._ax.arrow(x, y, *b, width=100, color="b")


def main():
    sensor = Sensor(50, 5)
    frames = sensor.truth.space

    fig, ax = plt.subplots()
    ax.set_ylim(-11000, 11000)
    ax.set_xlim(-11000, 11000)
    sensor.init(ax)
    ani = animation.FuncAnimation(fig, sensor.draw, frames, repeat=True, interval=10)
    plt.show()


if __name__ == '__main__':
    main()
