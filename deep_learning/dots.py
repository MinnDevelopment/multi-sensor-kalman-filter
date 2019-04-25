import matplotlib.pyplot as plt
from matplotlib import animation
from random import random


class Dot:
    def __init__(self):
        self.vectors = []
        self.dead = False
        self.won = False
        self.pos = (.0, .0)
        self.point = None
        self.champion = False
        for i in range(1000):
            self.vectors.append((random() * 20 - 10, random() * 20 - 10))

    def remove(self):
        if self.point:
            self.point.remove()
        self.pos = (0, 0)
        self.dead = False

    def draw(self, ax):
        if self.point is None:
            color = self.get_color()
            self.point, = ax.plot([0], [0], color)
        self.point.set_data(*self.pos)

    def get_color(self):
        color = 'ko'
        if self.champion:
            color = 'bo'
        if self.dead:
            color = 'ro'
        return color

    def update(self, t):
        if self.dead or self.won:
            return
        vec = self.vectors[t]
        x, y = (self.pos[0] + vec[0], self.pos[1] + vec[1])
        x, y = self.check_bounds(x, y)
        self.pos = (x, y)

    def check_bounds(self, x, y):
        if self.dead:
            return x, y
        if x < -100:
            x = -100
            self.dead = True
        elif x > 100:
            x = 100
            self.dead = True
        if y < 0:
            y = 0
            self.dead = True
        elif y > 200:
            y = 200
            self.dead = True
        if self.dead:
            self.point = None
        return x, y


class Population:
    def __init__(self, ax):
        self.ax = ax
        self.dots = []
        self.goal = (0, 190)
        self.goal_ptr = ax.plot(*self.goal, 'go')
        for i in range(100):
            self.dots.append(Dot())

    @staticmethod
    def _diff(a, b):
        y = abs(a[1] - b[1])
        x = abs(a[0] - a[0])
        return y / x if x != 0 else y

    def finish(self):
        champion = self.dots[0]
        for dot in self.dots:
            dot.champion = False
            if self._diff(self.goal, dot.pos) < self._diff(self.goal, champion):
                champion = dot

        champion.champion = True

    def reset(self):
        for dot in self.dots:
            dot.remove()

    def draw(self, t):
        self.update(t)
        for dot in self.dots:
            dot.draw(self.ax)

    def update(self, t):
        for dot in self.dots:
            dot.update(t)
            if self._diff(dot.pos, self.goal) < 2:
                dot.won = True


def main():
    fig, ax = plt.subplots()
    pop = Population(ax)
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 200)
    ax.text(50, 175, "Gen: %d" % 1, color="k")
    ani = animation.FuncAnimation(fig, pop.draw, frames=1000, interval=1)
    plt.show()


if __name__ == '__main__':
    main()
