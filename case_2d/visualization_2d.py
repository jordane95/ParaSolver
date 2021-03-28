import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from case_2d.numerical_2d import NumericalSolver


# numerical solution of two-dimensional time-independent flow
class NumSol4Vis(NumericalSolver):
    def __init__(self, s, w, beta, list_time):
        super().__init__(s, w, beta, list_time)
        self.list_length = None

    def calc_para_num(self):
        dim = 2
        delta_t = self.list_time[1]-self.list_time[0]
        steps = len(self.list_time)
        list_A = [self.A for i in range(steps)]
        # calculate tensor of deformation at each time
        trans = np.eye(dim)
        list_F = [trans]
        for i in range(steps):
            trans = self.calc_trans_mat(list_A[i], delta_t, list_F[i], dim)
            list_F.append(trans)
        # calculate the deformation parameters at each time
        list_value = []
        list_vector = []
        for i in range(len(list_F)):
            (eig_value, eig_vector) = self.calc_geo_parameters(list_F[i])
            list_value.append(eig_value)
            list_vector.append(eig_vector)
        self.list_length = np.array([[1/eig_values[0], 1/eig_values[1]] for eig_values in list_value])
        self.list_angle_num = self.calc_angle(list_vector)
        self.list_angle_num.pop(0)
        return self.list_length, self.list_angle_num


def simulation_2d(list_length, list_angle):
    fig = plt.figure()
    plt.ion()
    for (length, angle) in zip(list_length, list_angle):
        a = length[0]
        b = length[1]
        t = np.linspace(0, 2 * np.pi, 1000)
        list_x = a * np.cos(t)
        list_y = b * np.sin(t)
        cor_old = [[x, y] for (x, y) in zip(list_x, list_y)]
        rot = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
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
    fig = plt.figure()
    plt.ion()
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


def test():
    # get data
    numSolver = NumSol4Vis(s=0.5, w=1, beta=0, list_time=np.linspace(0, 5, 500))
    list_length, list_angle = numSolver.calc_para_num()

    # plot
    fill = True
    if fill:
        simulation_2d_fill(list_length, list_angle)
    else:
        simulation_2d(list_length, list_angle)


if __name__ == '__main__':
    test()