import numpy as np
import matplotlib.pyplot as plt
from case_3d.numerical_3d import NumericalSolver3D
from case_3d.analytical_3d import AnalyticalSolver3D


class Comparator:
    def __init__(self, s_1, s_2, w_z, list_time):
        self.s_1 = s_1
        self.s_2 = s_2
        self.w_z = w_z
        self.list_time = list_time

        self.anaSolver = AnalyticalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        self.numSolver = NumericalSolver3D(self.s_1, self.s_2, self.w_z, self.list_time)
        self.list_ratios_ana, self.list_angle_ana = self.anaSolver.calc_geo_para()
        self.list_ratios_num, self.list_angles_num = self.numSolver.calc_geo_para()

    def compare_ratio(self, dim=3):
        self.list_ratios_ana = np.array(self.list_ratios_ana)
        self.list_ratios_num = np.array(self.list_ratios_num)
        for i in range(dim):
            plt.subplot(1, 3, i+1)
            plt.plot(self.list_time, self.list_ratios_ana[:, i], color='r', label='ratio_ana')
            plt.plot(self.list_time, self.list_ratios_num[:, i], color='b', label='ratio_num')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('ratio')
        plt.show()
        return None

    def compare_angle(self):
        self.list_angle_ana = np.array(self.list_angle_ana)
        self.list_angles_num = np.array(self.list_angles_num)
        plt.plot(self.list_time, self.list_angle_ana, color='r', label='angle_ana')
        plt.plot(self.list_time, self.list_angles_num[:, 0, 0], color='b', label='angle_num')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('theta')
        plt.show()
        return None


def test():
    # At most 7s, otherwise, beyond the biggest number that the computer can represent.
    compartor = Comparator(s_1=2, s_2=1, w_z=0.75, list_time=np.linspace(0, 5, 500))
    compartor.compare_ratio()
    compartor.compare_angle()


if __name__ == '__main__':
    test()
