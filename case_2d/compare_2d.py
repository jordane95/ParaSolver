import numpy as np
import matplotlib.pyplot as plt
from case_2d.numerical_2d import NumericalSolver2D
from case_2d.analytical_2d import AnalyticalSolver2D


class AnaVsNum:
    def __init__(self, s, w, beta, list_t):
        self.s = s
        self.w = w
        self.beta = beta
        self.list_t = list_t

    def compare(self, log=False):
        list_time = self.list_t
        ana_solver = AnalyticalSolver2D(self.s, self.w, self.beta, self.list_t)
        num_solver = NumericalSolver2D(self.s, self.w, self.beta, self.list_t)
        list_ratio_ana, list_angle_ana, list_angle_ana_ = ana_solver.calc_geo_para()
        list_ratio_num, list_angle_num = num_solver.calc_geo_para()
        label = 'a/b'
        if log:
            list_ratio_ana = np.log(list_ratio_ana)
            list_ratio_num = np.log(list_ratio_num)
            label = 'log(a/b)'
        # compare ratio
        plt.subplot(1, 2, 1)
        plt.plot(list_time, list_ratio_num, color='b', ls='--', label='ratio_num')
        plt.plot(list_time, list_ratio_ana, color='r', ls=':', label='ratio_ana')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel(label)
        plt.title('evolution of ratio with time')
        # compare angle
        plt.subplot(1, 2, 2)
        plt.plot(list_time, list_angle_num, color='b', ls='--', label='angle_num')
        plt.plot(list_time, list_angle_ana, color='r', ls=':', label='angle_ana')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('theta')
        plt.title('evolution of angle with time')

        plt.show()
        return None


def test():
    comparator = AnaVsNum(s=0.5, w=1, beta=0, list_t=np.linspace(0, 10, 100))
    comparator.compare(log=False)


if __name__ == '__main__':
    test()
