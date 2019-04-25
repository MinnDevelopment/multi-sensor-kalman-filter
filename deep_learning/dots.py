import matplotlib.pyplot as plt
from matplotlib import animation
from random import random
from numpy import sqrt

def random_trajectory():
    return random() * 20 - 10

def dist(a, b):
    x = a[0]-b[0]
    y = a[1]-b[1]
    return sqrt(x**2 + y**2)

class Dot:
    def __init__(self):
        self.dead = False
        self.champion = False
        self.moves = []

    def next_move(self, t):
        if self.dead or t >= len(self.moves):
            return None
        return self.moves[t]

    def mutate(self):
        for move in self.moves:
            # small chance to change trajectory
            if random() < 0.1:
                move = (random_trajectory(), move[1])
            if random() < 0.1:
                move = (move[0], random_trajectory())
        # small chance to gain additional move
        if random() < 0.05:
            x = random_trajectory()
            y = random_trajectory()
            self.moves.append((x, y))

class Level:
    def __init__(self):
        self.goal = (0, 0)
        self.fig, self.ax = plt.subplots()
        self.dots = {}
        self.pos = {}
        self.obstacles = []

    def redraw_dots(self):
        for dot in self.get_dots():
            ptr = self.dots[dot]
            pos = self.pos[dot]
            ptr.remove()
            color = 'ko'
            if dot.champion:
                color = 'bo'
            elif dot.dead:
                color = 'ro'
            self.dots[dot], = self.ax.plot(*pos, color)

    def get_dots(self):
        return self.dots.keys()

    def get_position(self, dot):
        return self.pos[dot]

    def add_dots(self, amt):
        for i in range(amt):
            dot = Dot()
            self.dots[dot], = self.ax.plot(0, 0, 'ko')
            self.pos[dot] = (0, 0)

    def is_death(self, x, y):
        return x < -100 or x > 100 or y < 0 or y > 200

    @staticmethod
    def get_real_pos(x, y):
        if x < -100:
            x = -100
        elif x > 100:
            x = 100
        if y < 0:
            y = 0
        elif y > 200:
            y = 200
        return (x, y)

    def move_dot(self, dot, x, y):
        ptr = self.dots[dot]
        old_x, old_y = self.get_position(dot)
        new_x = old_x + x
        new_y = old_y + y
        if self.is_death(new_x, new_y):
            dot.dead = True
            ptr.remove()
            new_x, new_y = self.get_real_pos(new_x, new_y)
            self.dots[dot], = self.ax.plot(new_x, new_y, 'ro')
        else:
            ptr.set_data(new_x, new_y)
        self.pos[dot] = (new_x, new_y)

    def draw(self):
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(0, 200)
        self.ax.plot(*self.goal, 'go')
        plt.show()

        self.fig, self.ax = plt.subplots()
        self.redraw_dots()

class Population:
    def __init__(self, level, size):
        self.level = level
        self.size = size
        self.level.add_dots(size)

    def mutate(self, amt=100):
        for dot in self.level.get_dots():
            for i in range(100):
                dot.mutate()

    def find_champion(self):
        champ = None
        min_dist = 1000
        goal = self.level.goal
        for dot in self.level.get_dots():
            dot.champion = False
            pos = self.level.get_position(dot)
            distance = dist(goal, pos)
            if champ is None or distance < min_dist:
                champ = dot
                min_dist = distance

        if champ:
            champ.champion = True
        self.level.redraw_dots()
        return champ

    def next_move(self, t):
        moved = False
        for dot in self.level.get_dots():
            move = dot.next_move(t)
            if move:
                moved = True
                self.level.move_dot(dot, *move)

    def get_champion(self):
        for dot in self.level.get_dots():
            if dot.champion:
                return dot
        return None

def main():
    level = Level()
    level.goal = (0, 190)
    pop = Population(level, 100)
    pop.mutate(1000)

    ani = animation.FuncAnimation(level.fig, pop.next_move, 1000, interval=10)
    pop.find_champion()
    ani = animation.FuncAnimation(level.fig, pop.next_move, 1000, interval=10)



if __name__ == '__main__':
    main()
