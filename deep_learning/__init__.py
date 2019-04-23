import numpy as np
from random import random, randint


class Layer:
    def __init__(self, shape):
        self.weights = np.ndarray(shape)
        self.weights[:, :] = 0

    def compute(self, input_vector):
        return input_vector @ self.weights

    def mutate(self):
        while random() >= 0.5:
            row = randint(0, self.weights.shape[0] - 1)
            col = randint(0, self.weights.shape[1] - 1)
            self.weights[row][col] = random() * 2 - 1


class Network:
    def __init__(self, depth, input_size=2):
        self.bias = 0
        self.input_size = input_size
        self.layers = []
        shape = (input_size, input_size * 4)
        for i in range(depth - 1):
            self.layers.append(Layer(shape))
            shape = (shape[1], shape[1] // 2)
            if shape[0] == 1:
                break

        if shape[0] > 1:
            shape = (shape[0], 1)
            self.layers.append(Layer(shape))

        self.depth = len(self.layers)

    def compute(self, vector):
        for layer in self.layers:
            vector = layer.compute(vector)
            row = vector[0]
            for i in range(len(row)):
                row[i] = int(compute_node(row[i] - self.bias))

        return int(vector[0][0])

    def mutate(self):
        for layer in self.layers:
            layer.mutate()

    def clone(self):
        network = Network(self.depth, self.input_size)
        for layer in self.layers:
            network.layer = layer
        return network


def merge(n1, n2):
    network = Network(n1.depth, n1.input_size)
    pivot = randint(1, network.depth - 1)
    for i in range(network.depth - pivot):
        network.layers[i] = n1.layers[i]
    for i in range(pivot, network.depth):
        network.layers[i] = n2.layers[i]
    network.mutate()
    return network


def compute_node(result):
    return 1 if max(result, 0) > 0 else 0


def main():
    brains = []
    for i in range(100):
        brains.append(Network(10, 3))

    results = [0, 0]
    vec = np.array([[1, 0, 1]])
    for gen in range(10000):
        survivors = []
        for brain in brains:
            result = brain.compute(vec)
            if result == 1:
                survivors.append(brain)
            results[result] += 1

        children = []
        for brain in survivors:
            mate = randint(0, len(survivors) - 1)
            children.append(merge(brain, survivors[mate]))

        if len(children) == 0:
            brains[0].mutate()
            children.append(brains[0])
        while len(children) < 100:
            index = randint(0, len(children) - 1)
            child = children[index].clone()
            for m in range(randint(1, 10)):
                child.mutate()
            children.append(child)

        brains = children
            

    print("0 -> %s" % results[0])
    print("1 -> %s" % results[1])

    results = [0, 0]
    for brain in brains:
        res = brain.compute(vec)
        results[res] += 1

        
    print("0 -> %s" % results[0])
    print("1 -> %s" % results[1])


if __name__ == '__main__':
    main()
