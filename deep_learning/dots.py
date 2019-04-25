import matplotlib.pyplot as plt
import operator
from matplotlib import animation
from matplotlib.patches import Rectangle
from random import random, randint
from numpy import sqrt


def random_trajectory():
    if random() < 0.5:
        return random() * 5 - 10
    return random() * 5 + 5


def dist(a, b):
    x = a[0]-b[0]
    y = a[1]-b[1]
    return sqrt(x**2 + y**2)


class Dot:
    def __init__(self):
        self.dead = False
        self.champion = False
        self.win = False
        self.moves = []
        self.needed_moves = 0

    def reset(self):
        self.needed_moves = 0
        self.win = False
        self.dead = False

    def next_move(self, t):
        if self.dead or self.win or t >= len(self.moves):
            return None
        self.needed_moves += 1
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


class Obstacle:
    def __init__(self, a, b, c, d):
        self.pos = (a, b)
        self.width = abs(a - c)
        self.height = abs(b - d)

    def draw(self, ax):
        rec = Rectangle(self.pos, self.width, self.height)
        ax.add_patch(rec)

    def is_hit(self, x, y):
        return self.pos[0] <= x <= self.pos[0] + self.width and self.pos[1] <= y <= self.pos[1] + self.height


class Level:
    def __init__(self):
        self.goal = (0, 0)
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(0, 200)
        self.ax.set_xlim(-100, 100)
        self.ax.plot(0, 190, 'go')
        self.dots = {}
        self.pos = {}
        self.obstacles = []

    def add_obstacle(self, x1, y1, x2, y2):
        self.obstacles.append(Obstacle(x1, y1, x2, y2))

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.draw(self.ax)

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

    def add_dot(self, dot):
        ptr, = self.ax.plot(0, 0, 'ko')
        self.dots[dot] = ptr
        self.pos[dot] = (0, 0)

    def add_dots(self, amt):
        for i in range(amt):
            self.add_dot(Dot())

    def is_death(self, x, y):
        for o in self.obstacles:
            if o.is_hit(x, y):
                return True
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
        return x, y

    def del_dot(self, dot):
        ptr = self.dots.pop(dot)
        pos = self.pos.pop(dot)
        if ptr:
            ptr.remove()

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
        if dist(self.goal, (new_x, new_y)) < 2:
            dot.win = True
        self.pos[dot] = (new_x, new_y)

    def draw(self):
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(0, 200)
        self.ax.plot(*self.goal, 'go')
        plt.show()

        self.fig, self.ax = plt.subplots()
        self.redraw_dots()
        self.draw_obstacles()

    def reset(self):
        for dot in self.dots:
            self.pos[dot] = (0, 0)
        self.redraw_dots()


class Population:
    def __init__(self, level, size):
        self.gen = 1
        self.label = None
        self.level = level
        self.size = size
        self.level.add_dots(size)
        self.print_gen()

    def fitness(self, dot):
        pos = self.level.get_position(dot)
        fit = 100 if dot.win else 100 - dist(pos, self.level.goal)**2
        fit += 100 - dot.needed_moves
        return fit

    def print_gen(self):
        if self.label:
            self.label.remove()
        self.label = self.level.ax.text(50, 190, f'Gen {self.gen}')

    def selection(self):
        survivors = []
        for dot in self.level.get_dots():
            survivors.append((self.fitness(dot), dot))
        survivors.sort(key=operator.itemgetter(0))
        for i in range(len(survivors) // 2):
            fit, dot = survivors.pop(0)
            self.level.del_dot(dot)

    def repopulate(self):
        dots = list(self.level.get_dots())
        while len(self.level.get_dots()) < self.size:
            index = randint(0, len(dots) - 1)
            parent = dots[index]
            dot = Dot()
            dot.moves = list(parent.moves)
            self.level.add_dot(dot)

    def evolve(self):
        self.gen += 1
        self.print_gen()
        self.find_champion()
        self.selection()
        self.repopulate()
        self.mutate()
        self.level.reset()

    def mutate(self, amt=100):
        for dot in self.level.get_dots():
            for i in range(amt):
                dot.mutate()

    def find_champion(self):
        champ = None
        best_fit = -5000
        for dot in self.level.get_dots():
            dot.champion = False
            distance = self.fitness(dot)
            if champ is None or distance > best_fit:
                champ = dot
                best_fit = distance

        if champ:
            champ.champion = True
        return champ

    def next_move(self, t):
        for dot in self.level.get_dots():
            move = dot.next_move(t)
            if move:
                self.level.move_dot(dot, *move)

        return []

    def get_champion(self):
        for dot in self.level.get_dots():
            if dot.champion:
                return dot
        return None


def main():
    level = Level()
    level.goal = (0, 190)
    level.add_obstacle(-10, 75, 100, 90)
    level.add_obstacle(-100, 125, 10, 140)
    level.draw_obstacles()
    pop = Population(level, 100)
    pop.mutate(1000)

    ani = animation.FuncAnimation(level.fig, pop.next_move, 1000, interval=10)
    level.draw()
    for i in range(100):
        pop.evolve()
        ani = animation.FuncAnimation(level.fig, pop.next_move, 1000, interval=10)
        level.draw()


if __name__ == '__main__':
    main()
