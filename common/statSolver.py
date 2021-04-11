from common.paraSolver import ParaSolver
from common.fileIO import read_grad, read_position
from common.visualization import plot_position
import numpy as np
import matplotlib.pyplot as plt


class StatSolver:
    def __init__(self, path, num, delta_t, steps) -> None:
        self.num = num
        self.solver = []
        self.list_time = np.array([delta_t*i for i in range(steps)])
        for i in range(num):
            pos_name = path + "U_" + str(i) + ".txt"
            grad_name = path + "Grad_" + str(i) + ".txt"
            delta_t, list_position, list_v = read_position(pos_name)
            plot_position(list_position, dest=str(i)+".png")
            delta_t, list_grad = read_grad(grad_name)
            self.solver.append(ParaSolver(list_grad[:steps], self.list_time))

        self.lists_ratios = None
        self.lists_angles = None
        self.lists_coli = None

        self.list_ratios_avg = None
        self.list_angles_avg = None
        self.list_coli_avg = None

    def solve(self):
        self.lists_ratios, self.lists_angles = [], []
        self.lists_coli = []
        for para_solver in self.solver:
            list_ratios, list_angles = para_solver.calc_geo_para(normalize=True)
            self.lists_ratios.append(list_ratios)
            self.lists_angles.append(list_angles)
            list_coli = para_solver.calc_coli(normalize=True)
            self.lists_coli.append(list_coli)
        self.lists_ratios = np.array(self.lists_ratios)
        self.lists_angles = np.array(self.lists_angles)
        self.lists_coli = np.array(self.lists_coli)
        # import pprint
        # pprint.pprint(self.lists_ratios)
        return None

    def calc_avg(self):
        self.list_ratios_avg = np.sum(self.lists_ratios, axis=0) / self.num
        self.list_angles_avg = np.sum(self.lists_angles, axis=0) / self.num
        self.list_coli_avg = np.sum(self.lists_coli, axis=0) / self.num
        return None

    def plot_ratio_avg(self, log=False):
        list_ratios = np.array(self.list_ratios_avg)
        list_time = self.list_time
        if log:
            list_ratios = np.log(list_ratios)
        labels = ['a_1/a_0', 'a_2/a_1', 'a_2/a_0']
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.plot(list_time, list_ratios[:, i], label=labels[i])
            plt.xlabel('t')
            plt.ylabel('ratio')
            plt.legend()
        plt.show()
        return None

    def plot_angle_avg(self):
        list_angles = np.array(self.list_angles_avg)
        num = 0
        for i in range(3):
            for j in range(3):
                num += 1
                plt.subplot(3, 3, num)
                plt.ylim(0, np.pi / 2 + 0.1)
                plt.plot(self.list_time, list_angles[:, i, j],
                         label='angle(e_' + str(j + 1) + ', x_' + str(i + 1) + ')')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('theta')
        plt.show()
        return None

    def plot_coli_avg(self):
        list_coli = self.list_coli_avg
        num = 0
        for i in range(3):
            for j in range(3):
                num += 1
                plt.subplot(3, 3, num)
                plt.ylim(0, 1)
                plt.plot(self.list_time, [coli[i, j] for coli in list_coli],
                         label='<e_' + str(i + 1) + ', s_' + str(j + 1) + '>')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('inner product')
        plt.show()

    def calc_distribution(self):
        return None
