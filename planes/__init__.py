
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import norm

w = 9 / 600
a = 300**2 / 9

interval = np.arange(start=0, stop=1000 * pi, step=1)


def r(t):
    return [a * sin(2 * w*t), a * sin(2*w*t)]


def dt_r(t):
    return [150 * cos(w*t), 300 * cos(2*w*t)]


def dt_dt_r(t):
    return [9 * sin(w*t) / 4, 9 * sin(2*w*t)]


def t(t):
    s = 1 / norm(dt_r(t))
    return [s * cos(w*t), s * cos(2*w*t)]


def n(t):
    s = 1 / norm(dt_r(t))
    return [-s * cos(2*w*t), s * cos(w*t)]


def dt_dt_r_t(x):
    a = t(x)
    b = dt_dt_r(x)
    return [a[0] * b[0], a[1] * b[1]]


def dt_dt_r_n(t):
    a = n(t)
    b = dt_dt_r(t)
    return [a[0] * b[0], a[1] * b[1]]


def gen(fn):
    for t in interval:
        yield fn(t)


def main():
    fig, axes = plt.subplots()
    point, = axes.plot([0], [0], 'go')
    point.set_data(0, 0)
    #axes.grid()

    def init_frame():
        axes.set_ylim(-10, 10)
        axes.set_xlim(-3, 3)
        return point

    def run(data):
        t, y = data
        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()

        if t >= xmax:
            axes.set_xlim(xmin, t+1)
            axes.figure.canvas.draw()
        elif t <= xmin:
            axes.set_xlim(t-1, xmax)
            axes.figure.canvas.draw()
        if y >= ymax:
            axes.set_ylim(ymin, y+1)
            axes.figure.canvas.draw()
        elif y <= ymin:
            axes.set_ylim(y-1, ymax)
            axes.figure.canvas.draw()

        point.set_data(t, y)
        return point

    ani = animation.FuncAnimation(fig, run, gen(dt_dt_r), init_func=init_frame, interval=10)
    plt.show()


if __name__ == '__main__':
    main()