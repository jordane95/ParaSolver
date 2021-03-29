import numpy as np


def read_grad(filename):
    with open(filename, 'r') as f:
        a = f.readline()
        delta = float(a)
        list_grad = []
        for line in f.readlines():
            grad = [float(i) for i in line.strip().split(' ')]
            grad = np.array(grad).reshape(3, 3)
            list_grad.append(grad)
    return delta, list_grad


def read_position(filename):
    with open(filename, 'r') as f:
        a = f.readline()
        delta = float(a)
        list_position = []
        list_velocity = []
        for line in f.readlines():
            array = [float(i) for i in line.strip().split(' ')]
            position = array[:3]
            velocity = array[3:]
            list_position.append(position)
            list_velocity.append(velocity)
    return delta, np.array(list_position), np.array(list_velocity)


def test_grad():
    filename = 'gradU.txt'
    delta, list_grad = read_grad(filename)
    print(delta, type(delta))
    print(list_grad)


def test_position():
    filename = 'Utr.txt'
    delta, list_position, list_velocity = read_position(filename)
    print(list_position)


if __name__ == '__main__':
    test_position()
