import numpy as np
import matplotlib.pyplot as plt


# numerical solution of two-dimensional time-independent flow
class NumericalSolver2D:
    def __init__(self, s, w, beta, list_time):
        self.A = self.make_grad_tensor(s, w, beta)
        self.list_time = list_time
        self.list_ratio = None
        self.list_angle = None

    @staticmethod
    def make_grad_tensor(s, w, beta):
        sym = np.array([[s * np.cos(2 * beta), s * np.sin(2 * beta)],
                        [s * np.sin(2 * beta), -s * np.cos(2 * beta)]])
        anti_sym = np.array([[0, -w],
                             [w, 0]])
        return sym + anti_sym

    @staticmethod
    def calc_F_next(A, delta_t, F_prec, dim):
        D = np.eye(dim) - (delta_t / 2) * A
        E = np.eye(dim) + (delta_t / 2) * A
        F = np.dot(np.dot(np.linalg.pinv(D), E), F_prec)
        return F

    @staticmethod
    def decompose(F):
        F_inv = np.linalg.pinv(F)
        G = np.dot(F_inv.transpose(), F_inv)
        params = np.linalg.eig(G)
        return params

    # calculate the ratio and angle for two dimensional flow
    # from the eig values and eig vectors
    @staticmethod
    def calc_ratio(list_value):
        list_ratio = [np.sqrt(eig_values.max()/eig_values.min()) for eig_values in list_value]
        return list_ratio

    @staticmethod
    def calc_angle(list_vectors):
        list_angle = []
        for rot_mat in list_vectors:
            angle = np.arccos(rot_mat[0][0]) # if the axis inverse, the angle too
            if np.abs(angle) > np.pi/2:
                angle = angle-np.pi
            list_angle.append(angle)
        return list_angle

    def calc_eig_para(self):
        dim = 2
        delta_t = self.list_time[1] - self.list_time[0]
        steps = len(self.list_time)
        list_A = [self.A for _ in range(steps)]
        # calculate tensor of deformation at each time
        trans = np.eye(dim)
        list_F = [trans]
        for i in range(steps):
            trans = self.calc_F_next(list_A[i], delta_t, list_F[i], dim)
            list_F.append(trans)
        # calculate the deformation parameters at each time
        list_value = []
        list_vector = []
        for F in list_F:
            eig_value, eig_vector = self.decompose(F)
            list_value.append(eig_value)
            list_vector.append(eig_vector)
        return list_value, list_vector

    def calc_geo_para(self):
        list_value, list_vector = self.calc_eig_para()
        self.list_ratio = self.calc_ratio(list_value)
        self.list_angle = self.calc_angle(list_vector)
        self.list_ratio.pop(0)
        self.list_angle.pop(0)
        return self.list_ratio, self.list_angle

    def plot_ratio_num(self):
        plt.plot(self.list_time, self.list_ratio, label='ratio')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('ratio')
        plt.show()

    def plot_angle_num(self):
        plt.plot(self.list_time, self.list_angle, label='angle')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('angle')
        plt.show()
