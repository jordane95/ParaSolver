import numpy as np
import matplotlib.pyplot as plt
import pprint


# numerical solution of two-dimensional time-independent flow
class NumericalSolver2D:
    def __init__(self, s, w, beta, list_time, force_sort=None):
        sym = np.array([[s * np.cos(2 * beta), s * np.sin(2 * beta)],
                        [s * np.sin(2 * beta), -s * np.cos(2 * beta)]])
        anti_sym = np.array([[0, -w],
                             [w, 0]])
        self.A = sym + anti_sym

        # indicator
        self.sort = True if s**2-w**2 >= 0 else False
        if force_sort:
            self.sort = True

        self.list_time = list_time
        self.list_ratio = None
        self.list_angle = None

    @staticmethod
    def calc_F_next(A, delta_t, F_prec, dim):
        D = np.eye(dim) - (delta_t / 2) * A
        E = np.eye(dim) + (delta_t / 2) * A
        F = np.dot(np.dot(np.linalg.pinv(D), E), F_prec)
        return F

    @staticmethod
    def decompose(G, params_prev=None):
        params_next = np.linalg.eig(G)
        if params_prev is not None:
            eig_values_prev, eig_vectors_prev = params_prev
            eig_values_next, eig_vectors_next = params_next
            sim = np.abs(eig_vectors_next.T @ eig_vectors_prev)
            sort_idx = np.argmax(sim, axis=1)
            eig_values_next = eig_values_next[sort_idx]
            eig_vectors_next = (eig_vectors_next.T[sort_idx]).T
            params_next = (eig_values_next, eig_vectors_next)
        return params_next

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
        params = (np.array([1, 1]), np.eye(2))
        for F in list_F:
            F_inv = np.linalg.pinv(F)
            G = np.dot(F_inv.transpose(), F_inv)
            if self.sort:
                params = self.decompose(G, params)
            else:
                params = self.decompose(G)
            eig_value, eig_vector = params
            list_value.append(eig_value)
            list_vector.append(eig_vector)
        return list_value, list_vector

    # calculate the ratio and angle for two dimensional flow
    # from the eig values and eig vectors
    @staticmethod
    def calc_ratio(list_value, sort=True):
        if sort:
            list_ratio = [np.sqrt(eig_values[1]/eig_values[0]) for eig_values in list_value]
        else:
            list_ratio = [np.sqrt(eig_values.max() / eig_values.min()) for eig_values in list_value]
        return list_ratio

    @staticmethod
    def calc_angle(list_vectors):
        list_angle = []
        for rot_mat in list_vectors:
            angle = np.arccos(rot_mat[0][0])
            if angle > np.pi/2: angle -= np.pi
            if angle < -np.pi/2: angle += np.pi
            list_angle.append(angle)
        return list_angle

    def calc_geo_para(self):
        list_value, list_vector = self.calc_eig_para()
        self.list_ratio = self.calc_ratio(list_value, self.sort)
        self.list_angle = self.calc_angle(list_vector)
        self.list_ratio.pop(0)
        self.list_angle.pop(0)
        return self.list_ratio, self.list_angle

    def plot_ratio_num(self, log=True):
        label = 'a_1/a_2'
        if log:
            plt.plot(self.list_time, np.log(self.list_ratio), label='ratio_num')
            label = 'log(a_1/a_2)'
        else:
            plt.plot(self.list_time, self.list_ratio, label='ratio_num')
        plt.xlabel('time')
        plt.ylabel(label)
        plt.legend()
        plt.show()

    def plot_angle_num(self):
        plt.plot(self.list_time, self.list_angle, label='angle_num')
        plt.xlabel('time')
        plt.ylabel('theta')
        plt.legend()
        plt.show()


def test():
    num_solver = NumericalSolver2D(s=0.5, w=1, beta=0, list_time=np.linspace(0, 20, 1000), force_sort=True)
    num_solver.calc_eig_para()
    num_solver.calc_geo_para()
    num_solver.plot_ratio_num()


if __name__ == '__main__':
    test()