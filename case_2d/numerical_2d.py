import numpy as np
import matplotlib.pyplot as plt
import pprint


# numerical solution of two-dimensional time-independent flow
class NumericalSolver2D:
    def __init__(self, s, w, beta, list_time):
        sym = np.array([[s * np.cos(2 * beta), s * np.sin(2 * beta)],
                        [s * np.sin(2 * beta), -s * np.cos(2 * beta)]])
        anti_sym = np.array([[0, -w],
                             [w, 0]])
        self.A = sym + anti_sym

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
    def decompose(F, params_prev=None):
        F_inv = np.linalg.pinv(F)
        G = np.dot(F_inv.transpose(), F_inv)
        params_next = np.linalg.eig(G)
        print()
        print("Before sort:")
        pprint.pprint(params_next)
        if params_prev is not None:
            eig_values_prev, eig_vectors_prev = params_prev
            eig_values_next, eig_vectors_next = params_next
            sim = np.abs(eig_vectors_next.T @ eig_vectors_prev)
            # print("Similarity Matrix:\n", sim)
            sort_idx = np.argmax(sim, axis=1)
            print("Sort idx: ", sort_idx)
            eig_values_next = eig_values_next[sort_idx]
            eig_vectors_next = (eig_vectors_next.T[sort_idx]).T
            # for j in range(eig_vectors_next.shape[1]):
            #     eig_vectors = eig_vectors_next[:, j]
            #     if eig_vectors[1] < 0: # sin<0
            #         eig_vectors_next[:, j] = -eig_vectors
            if sort_idx[0] == 1: print("=" * 10 + "Inversion here" + "=" * 80)
            params_next = (eig_values_next, eig_vectors_next)
        print("After sort: ")
        pprint.pprint(params_next)
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
            params = self.decompose(F, params)
            eig_value, eig_vector = params
            list_value.append(eig_value)
            list_vector.append(eig_vector)
        return list_value, list_vector

    # calculate the ratio and angle for two dimensional flow
    # from the eig values and eig vectors
    @staticmethod
    def calc_ratio(list_value):
        # list_ratio = [np.sqrt(eig_values.max() / eig_values.min()) for eig_values in list_value]
        list_ratio = [np.sqrt(eig_values[1]/eig_values[0]) for eig_values in list_value]
        return list_ratio

    @staticmethod
    def calc_angle(list_vectors):
        list_angle = []
        for rot_mat in list_vectors:
            # pprint.pprint(rot_mat)
            # if rot_mat[1][0] >= 0: angle = np.arccos(rot_mat[0][0])
            # if rot_mat[1][0] < 0: angle = np.arccos(-rot_mat[0][0])
            angle = np.arccos(rot_mat[0][0])
            # if angle > np.pi/2: angle -= np.pi
            # if angle < -np.pi/2: angle += np.pi
            # print(angle)
            list_angle.append(angle)
        return list_angle

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
