import matplotlib.pyplot as plt
import numpy as np


def _plot_ellipse_3d(lengths, trans_mat, ax):
    (a, b, c) = tuple(lengths)
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2 * np.pi, 20)
    list_x = a * np.outer(np.sin(theta), np.sin(phi))
    list_y = b * np.outer(np.sin(theta), np.cos(phi))
    list_z = c * np.outer(np.cos(theta), np.ones_like(phi))
    list_cor_old = [[x, y, z] for (x, y, z) in zip(list_x, list_y, list_z)]
    list_cor_new = np.array([np.dot(trans_mat, cor) for cor in list_cor_old])
    plt.cla()
    # ax.plot_surface(list_cor_new[:, 0], list_cor_new[:, 1], list_cor_new[:, 2], color='b')
    ax.plot_wireframe(list_cor_new[:, 0], list_cor_new[:, 1], list_cor_new[:, 2], color='r')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.pause(0.05)


def simulation_3d(list_length, list_vectors):
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.ion()
    time = 0
    for (lengths, trans_mat) in zip(list_length, list_vectors):
        print("At time %d" % time)
        _plot_ellipse_3d(lengths, trans_mat, ax)
        time += 1
    print("Finished!")
    plt.ioff()
    plt.show()


# plot the trajectory of a particle
def plot_position(list_position):
    print("Plotting trajectoiry...")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(list_position[:, 0], list_position[:, 1], list_position[:, 2],
               c='b', marker='o', s=1)
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    print('Finished')
    plt.show()


# the following code is to test the previous functions
def test_simulation():
    from case_2d.numerical_2d import NumericalSolver2D
    temp = NumericalSolver2D.make_grad_tensor(s=.5, w=1, beta=0)
    A = np.zeros((3, 3))
    A[:2, :2] = temp
    list_time = np.linspace(0, 5, 200)
    list_A = [A for _ in list_time]

    from common.paraSolver import ParaSolver
    solver = ParaSolver(list_A=list_A, list_time=list_time)
    list_eig_values, list_eig_vectors = solver.calc_eig_para()
    list_length = np.sqrt(1 / np.array(list_eig_values))
    print("Calculation Done!")

    # animation
    simulation_3d(list_length, list_eig_vectors)


def test_position():
    from common.file_io import read_position
    filename = 'Utr.txt'
    delta, list_position, list_velocity = read_position(filename)
    plot_position(np.array(list_position))


if __name__ == '__main__':
    test_simulation()
