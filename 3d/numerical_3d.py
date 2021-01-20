import numpy as np
import matplotlib.pyplot as plt


class NumericalSolver3D:
    def __init__(self, s_1, s_2, w_z, list_time):
        self.A = self.make_grad_tensor(s_1, s_2, w_z)
        self.list_time = list_time
        self.eig_values = None
        self.eig_vectors = None

        self.list_ratio_xy = None
        self.list_ratio_yz = None
        self.list_ratio_xz = None

    @staticmethod
    def make_grad_tensor(s_1, s_2, w_z):
        A = np.array([[s_1, -w_z, 0],
                      [w_z, s_2, 0],
                      [0, 0, -(s_1+s_2)]])
        return A

    # numerical solution
    @staticmethod
    def calc_trans_mat(A, delta_t, F_prec, dim):
        D = np.eye(dim) - (delta_t / 2) * A
        E = np.eye(dim) + (delta_t / 2) * A
        F = np.dot(np.dot(np.linalg.pinv(D), E), F_prec)
        return F

    @staticmethod
    def calc_geo_parameters(F):
        F_inv = np.linalg.pinv(F)
        G = np.dot(F_inv.transpose(), F_inv)
        parameters = np.linalg.eig(G)
        return parameters

    def calc_eig_value_vector(self):
        dim = 3
        delta_t = self.list_time[1]-self.list_time[0]
        steps = len(self.list_time)
        # make the list of grad_u tensor
        list_A = []
        for i in range(steps):
            list_A.append(self.A)
        # calculate tensor of deformation at each time
        trans = np.eye(dim)
        list_F = [trans]
        for i in range(steps):
            trans = self.calc_trans_mat(list_A[i], delta_t, list_F[i], dim)
            list_F.append(trans)
        list_F.pop(0)
        # calculate the deformation parameters at each time
        list_value = []
        list_vector = []
        for F in list_F:
            (eig_value, eig_vector) = self.calc_geo_parameters(F)
            list_value.append(eig_value)
            list_vector.append(eig_vector)
        self.eig_values = list_value
        self.eig_vectors = list_vector
        # print(list_vector)
        return list_value, list_vector

    def calc_ratio(self):
        self.calc_eig_value_vector()
        list_values = self.eig_values
        list_ratio_xy = []
        list_ratio_yz = []
        list_ratio_xz = []
        for i in range(len(list_values)):
            axes = list_values[i]
            ratio_xy = np.sqrt(axes[1]/axes[0])
            ratio_yz = np.sqrt(axes[2]/axes[1])
            ratio_xz = np.sqrt(axes[2]/axes[0])
            list_ratio_xy.append(ratio_xy)
            list_ratio_yz.append(ratio_yz)
            list_ratio_xz.append(ratio_xz)
        self.list_ratio_xy = list_ratio_xy
        self.list_ratio_yz = list_ratio_yz
        self.list_ratio_xz = list_ratio_xz
        return list_ratio_xy, list_ratio_yz, list_ratio_xz

    def calc_angle(self):
        self.calc_eig_value_vector()
        list_angle = []
        for matrix in self.eig_vectors:
            theta = np.arccos(matrix[0][0])
            if np.abs(theta) > np.pi/2:
                theta = theta-np.pi
            list_angle.append(theta)
        return list_angle

    def plot_ratio(self):
        self.calc_ratio()
        plt.subplot(1, 3, 1)
        plt.plot(self.list_time, self.list_ratio_xy, label='a1/a2')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(self.list_time, self.list_ratio_yz, label='a2/a3')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(self.list_time, self.list_ratio_xz, label='a1/a3')
        plt.legend()
        plt.show()
        return None

'''
numSolver3 = NumericalSolver3D(s_1=2, s_2=1, w_z=0.1, list_time=np.linspace(0, 1, 100))
(eig_values, eig_vectors) = numSolver3.calc_eig_value_vector()
for i in range(len(eig_values)):
    print(eig_values[i])
numSolver3.plot_ratio()
'''