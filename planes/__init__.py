
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

    def plot(self):
        fig, ax = plt.subplots()
        t, d = ([self.position(t)[0] for t in self.space], [self.position(t)[1] for t in self.space])
        ax.plot(t, d, 'black')
        point, = ax.plot([0], [0], 'go')
        point.set_data(0, 0)

        def redraw(x, y):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            if x < xmin:
                ax.set_xlim(x-1, xmax)
                ax.figure.canvas.draw()
            elif x > xmax:
                ax.set_xlim(xmin, x+1)
                ax.figure.canvas.draw()
            if y < ymin:
                ax.set_ylim(y-1, ymax)
                ax.figure.canvas.draw()
            elif y > ymax:
                ax.set_ylim(ymin, y+1)
                ax.figure.canvas.draw()

        def init():
            ax.set_ylim(-11000, 11000)
            ax.set_xlim(-11000, 11000)
            return point

        def frame(data):
            x, y = data
            redraw(x, y)
            point.set_data(x, y)
            return point

        movement = ( self.position(t) for t in self.space )
        ani = animation.FuncAnimation(fig, frame, movement, init_func=init, interval=100)
        plt.show()

    def _x(self, t):
        return self.a * sin(self.w * t)

    def _y(self, t):
        return self.a * sin(2 * self.w * t)

    def position(self, t):
        return np.array([[ self._x(t) ],
                         [ self._y(t) ]])

def main():
    truth = GroundTruth(300, 9)
    truth.plot()

if __name__ == '__main__':
    main()
