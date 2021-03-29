import numpy as np
from case_3d.numerical_3d import NumericalSolver3D
from common.visualization import simulation_3d


def test():
    # get data
    solver = NumericalSolver3D(s_1=2, s_2=1, w_z=0.75, list_time=np.linspace(0, 1, 100))
    solver.calc_eig_para()
    solver.calc_geo_para()
    list_length = np.sqrt(1/np.array(solver.list_eig_values))
    print("Calculation Done")
    # plot
    print('Plotting...')
    simulation_3d(list_length, solver.list_eig_vectors)


if __name__ == '__main__':
    test()