import numpy as np
import matplotlib.pyplot as plt


def simulation_2d(list_length, list_vectors):
    plt.ion()
    print("Plotting...")
    for (length, rot) in zip(list_length, list_vectors):
        a = length[0]
        b = length[1]
        t = np.linspace(0, 2 * np.pi, 1000)
        list_x = a * np.cos(t)
        list_y = b * np.sin(t)
        cor_old = [[x, y] for (x, y) in zip(list_x, list_y)]
        list_cor_new = np.array([np.dot(rot, cor) for cor in cor_old])

        plt.cla()
        plt.plot(list_cor_new[:, 0], list_cor_new[:, 1])
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.pause(0.01)
    print("Finished")
    plt.ioff()
    plt.show()


def simulation_2d_fill(list_length, list_angle):
    from matplotlib.patches import Ellipse
    fig = plt.figure()
    plt.ion()
    print("Plotting...")
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    for length, angle in zip(list_length, list_angle):
        plt.cla()
        e = Ellipse(xy=(0, 0), width=length[0]*2, height=length[1]*2, angle=angle*180/np.pi)
        ax.add_artist(e)
        e.set_facecolor("blue")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.pause(0.01)
    print("Finished")
    plt.ioff()
    plt.show()


def test(fill=True):
    from case_2d.numerical_2d import NumericalSolver2D
    num_solver_2d = NumericalSolver2D(s=0.5, w=1, beta=0, list_time=np.linspace(0, 5, 500))
    list_values, list_vectors = num_solver_2d.calc_eig_para()
    list_ratio, list_angle = num_solver_2d.calc_geo_para()
    list_length = np.array([[1/eig_values[0], 1/eig_values[1]] for eig_values in list_values])
    if fill:
        simulation_2d_fill(list_length, list_angle)
    else:
        simulation_2d(list_length, list_vectors)


if __name__ == '__main__':
    test()
