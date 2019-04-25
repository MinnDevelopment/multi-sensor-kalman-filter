import matplotlib.pyplot as plt
from matplotlib import animation
from random import random

def random_trajectory():
    return random() * 20 - 10

class Dot:
    def __init__(self):
        self.dead = False
        self.champion = False
        self.moves = []
        self.move_count = 0

    def next_move(self):
        if self.dead or len(self.moves) <= self.move_count:
            return None
        move = self.moves[self.move_count]
        self.move_count += 1
        return move

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
        self.obstacles = []

    def get_dots(self):
        return self.dots.keys()

    def get_position(self, dot):
        return self.dots[dot].get_data()

    def add_dots(self, amt):
        for i in range(amt):
            dot = Dot()
            self.dots[dot], = self.ax.plot(0, 0, 'ko')

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
        old_x, old_y = ptr.get_data()
        new_x = old_x + x
        new_y = old_y + y
        if self.is_death(new_x, new_y):
            dot.dead = True
            ptr.remove()
            new_x, new_y = self.get_real_pos(new_x, new_y)
            self.dots[dot], = self.ax.plot(new_x, new_y, 'ro')
        else:
            ptr.set_data(new_x, new_y)

    def draw(self):
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(0, 200)
        self.ax.plot(*self.goal, 'go')
        plt.show()

        self.fig, self.ax = plt.subplots()
        for dot in self.get_dots():
            old_pos = self.dots[dot].get_data()
            if dot.dead:
                self.dots[dot], = self.ax.plot(old_pos, 'ro')
            elif dot.champion:
                self.dots[dot], = self.ax.plot(old_pos, 'bo')
            else:
                self.dots[dot], = self.ax.plot(old_pos, 'ko')

class Population:
    def __init__(self, level):
        self.level = level

    def find_champion(self):
        champ = None
        goal = self.level.goal
        for dot in self.level.get_dots():
            dot.champion = False
            if champ is None or distance(goal, self.level.get_position(dot)) < distance(goal, self.level.get_position(champ)):
                champ = dot

        if champ:
            champ.champion = True
        return champ

    def get_champion(self):
        for dot in self.level.get_dots():
            if dot.champion:
                return dot
        return None

def main():
    level = Level()
    level.goal = (0, 190)
    level.add_dots(5)
    for dot in level.get_dots():
        for i in range(100):
            dot.mutate()
    for dot in level.get_dots():
        move = dot.next_move()
        while move:
            level.move_dot(dot, *move)
            move = dot.next_move()

    level.draw()



if __name__ == '__main__':
    main()
