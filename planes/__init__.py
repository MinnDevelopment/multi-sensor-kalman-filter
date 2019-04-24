
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import norm

class GroundTruth:
    def __init__(self, v, q):
        self.v = v
        self.q = q
        self.w = q / 2*v
        self.a = v**2 / q
        self.interval = 2 / self.w * pi
        self.space = np.linspace(0, self.interval, num=500)
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


def main():
    truth = GroundTruth(300, 9)
    truth.plot()


if __name__ == '__main__':
    main()
