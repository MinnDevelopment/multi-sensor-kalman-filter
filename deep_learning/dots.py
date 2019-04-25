import matplotlib.pyplot as plt
from matplotlib import animation
from random import random

class Dot:
    def __init__(self):
        self.dead = False
        self.champion = False
        self.moves = []
        self.move_count = 0

    def next_move(self):
        if len(self.moves) <= self.move_count:
            return (0, 0)
        move = self.moves[self.move_count]
        self.move_count += 1
        return move

class Level:
    def __init__(self):
        self.goal = (0, 0)
        self.fig, self.ax = plt.subplots()
        self.dots = {}
        self.obstacles = []

    def get_dots(self):
        return self.dots.keys()

    def add_dots(self, amt):
        for i in range(amt):
            dot = Dot()
            self.dots[dot], = self.ax.plot(0, 0, 'ko')

    def move_dot(self, dot, x, y):
        ptr = self.dots[dot]
        old_x, old_y = ptr.get_data()
        ptr.set_data(old_x + x, old_y + y)

    def draw(self):
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(0, 200)
        self.ax.plot(*self.goal, 'go')
        plt.show()

        self.fig, self.ax = plt.subplots()


def main():
    level = Level()
    level.goal = (0, 190)
    level.add_dots(5)
    x = 1
    y = 50
    for dot in level.get_dots():
        dot.moves.append((x, y))
        x += 5
    level.get_dots().__iter__().__next__().moves.append((-50, 5))
    for dot in level.get_dots():
        move = dot.next_move()
        while move != (0, 0):
            level.move_dot(dot, *move)
            move = dot.next_move()

    level.draw()


if __name__ == '__main__':
    main()
