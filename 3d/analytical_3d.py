import numpy as np
import matplotlib.pyplot as plt


class AnalyticalSolver3D:
    def __init__(self, s_1, s_2, w_z, list_time):
        self.s_1 = s_1
        self.s_2 = s_2
        self.w_z = w_z
        self.list_time = list_time
        self.list_ratio_xy = None
        self.list_ratio_yz = None
        self.list_ratio_xz = None
        self.list_angle = None
        return None

    # when delta > 0
    def delta_sup_zero(self):
        tmp = (self.s_1-self.s_2)**2
        b = np.sqrt(np.abs(tmp-4*self.w_z**2)) / 2
        list_ratio_xy = []
        list_ratio_yz = []
        list_ratio_xz = []
        list_angle = []
        for t in self.list_time:
            # calculate ratio
            p = 2*(np.cosh(b*t)**2 + ((tmp+4*self.w_z**2) * np.sinh(b*t)**2 / (tmp-4*self.w_z**2)))
            q = np.sqrt(4 * tmp / b**2 * np.sinh(b*t)**2 * (np.cosh(b*t)**2 + self.w_z**2 / b**2 * np.sinh(b*t)**2))
            print('p=', p, 'q=', q, 'p-q=', p-q)
            ratio_xy = np.sqrt((p+q)/(p-q))
            ratio_xz = np.sqrt(2*np.exp(3*(self.s_1+self.s_2)*t)/(p-q))
            ratio_yz = np.sqrt(2*np.exp(3*(self.s_1+self.s_2)*t)/(p+q))
            list_ratio_xy.append(ratio_xy)
            list_ratio_yz.append(ratio_yz)
            list_ratio_xz.append(ratio_xz)
            # calculate angle
            tan = -self.w_z*np.tanh(b*t)/b
            theta = np.arctan(tan)/2
            list_angle.append(theta)
        return list_ratio_xy, list_ratio_yz, list_ratio_xz, list_angle

    def delta_equ_zero(self):
        s_1 = self.s_1
        s_2 = self.s_2
        w_z = self.w_z
        tmp = (s_1-s_2)**2
        list_ratio_xy = []
        list_ratio_yz = []
        list_ratio_xz = []
        list_angle = []
        for t in self.list_time:
            # compute ratio
            #p = 2+tmp*t**2
            #q = np.sqrt(tmp * (1 + tmp * t**2) * t**2)
            det = 1 - ((s_1-s_2)*t/2)**2 + (w_z*t)**2
            p = 2*(1+((s_1-s_2)*t/2)**2+(w_z*t)**2)/det
            q = 2*np.abs(s_1-s_2)*t*np.sqrt(1+(w_z*t)**2)/det
            print('p=', p, 'q=', q, 'p-q=', p - q)
            ratio_xy = np.sqrt((p+q)/(p-q))
            ratio_xz = np.sqrt(2 * np.exp(3*(s_1+s_2)*t) * det / (p-q))
            ratio_yz = np.sqrt(2 * np.exp(3*(s_1+s_2)*t) * det / (p+q))
            list_ratio_xy.append(ratio_xy)
            list_ratio_xz.append(ratio_xz)
            list_ratio_yz.append(ratio_yz)
            # compute angle
            theta = np.arctan(-self.w_z*t)/2
            list_angle.append(theta)
        return list_ratio_xy, list_ratio_yz, list_ratio_xz, list_angle

    # when delta < 0
    def delta_inf_zero(self):
        tmp = (self.s_1 - self.s_2)**2
        b = np.sqrt(np.abs(tmp - 4 * self.w_z ** 2)) / 2
        list_ratio_xy = []
        list_ratio_xz = []
        list_ratio_yz = []
        list_angle = []
        for t in self.list_time:
            # calculate ratio
            p = 2*(np.cos(b*t)**2 + (tmp+4*self.w_z**2) * np.sin(b*t)**2 / (tmp-4*self.w_z**2))
            q = np.sqrt(4 * tmp * np.sin(b*t)**2 * (np.cos(b*t)**2 + self.w_z**2 * np.sin(b*t)**2 / b**2))/b
            print('p=', p, 'q=', q, 'p-q=', p-q)
            ratio_xy = np.sqrt((p+q)/np.abs(p-q))
            ratio_xz = np.sqrt(2*np.exp(3*(self.s_1+self.s_2)*t)/(p+q))
            ratio_yz = np.sqrt(2*np.exp(3*(self.s_1+self.s_2)*t)/np.abs(p-q))
            list_ratio_xy.append(ratio_xy)
            list_ratio_xz.append(ratio_xz)
            list_ratio_yz.append(ratio_yz)
            # calculate angle
            theta = np.arctan(-self.w_z/b*np.tan(b*t))/2
            list_angle.append(theta)
        return list_ratio_xy, list_ratio_yz, list_ratio_xz, list_angle

    def calc_para_ana(self):
        delta = (self.s_1-self.s_2)**2-4*self.w_z**2
        if delta > 0:
            (list_ratio_xy, list_ratio_yz, list_ratio_xz, list_angle) = self.delta_sup_zero()
        elif delta == 0:
            (list_ratio_xy, list_ratio_yz, list_ratio_xz, list_angle) = self.delta_equ_zero()
        else:
            (list_ratio_xy, list_ratio_yz, list_ratio_xz, list_angle) = self.delta_inf_zero()
        self.list_ratio_xy = list_ratio_xy
        self.list_ratio_yz = list_ratio_yz
        self.list_ratio_xz = list_ratio_xz
        self.list_angle = list_angle
        return list_ratio_xy, list_ratio_yz, list_ratio_xz, list_angle

    def plot_ratio(self):
        self.calc_para_ana()
        plt.subplot(1, 3, 1)
        plt.plot(self.list_time, self.list_ratio_xy, label='a1/a2')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(self.list_time, self.list_ratio_yz, label='a2/a3')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(self.list_time, self.list_ratio_xz, label='a3/a1')
        plt.legend()
        plt.show()
        return None

'''
anaSolver3 = AnalyticalSolver3D(s_1=2, s_2=1, w_z=0.1, list_time=np.linspace(0, 1, 100))
anaSolver3.plot_ratio()
'''