import numpy as np
import matplotlib.pyplot as plt
from case_2d.numerical_2d import NumericalSolver
from case_2d.analytical_2d import AnalyticalSolver


class AnaVsNum:
    def __init__(self, s, w, beta, list_t):
        self.s = s
        self.w = w
        self.beta = beta
        self.list_t = list_t

    def compare(self):
        list_time = self.list_t
        ana_solver = AnalyticalSolver(self.s, self.w, self.beta, self.list_t)
        num_solver = NumericalSolver(self.s, self.w, self.beta, self.list_t)
        (list_ratio_ana, list_angle_ana) = ana_solver.calc_para_ana()
        (list_ratio_num, list_angle_num) = num_solver.calc_para_num()
        # compare ratio
        plt.subplot(1, 2, 1)
        plt.plot(list_time, list_ratio_num, color='b', label='ratio_num')
        plt.plot(list_time, list_ratio_ana, color='r', label='ratio_ana')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('ratio')
        plt.title('evolution of ratio with time')
        # compare angle
        plt.subplot(1, 2, 2)
        plt.plot(list_time, list_angle_num, color='b', label='angle_num')
        plt.plot(list_time, list_angle_ana, color='r', label='angle_ana')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('angle')
        plt.title('evolution of angle with time')

        plt.show()
        return None


def test():
    comparator = AnaVsNum(s=1, w=1, beta=0, list_t=np.linspace(0, 5, 500))
    comparator.compare()


if __name__ == '__main__':
    test()