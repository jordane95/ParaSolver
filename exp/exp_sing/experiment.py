from common.fileIO import read_grad, get_position
from common.paraSolver import ParaSolver
from common.visualization import simulation_3d, plot_position
import numpy as np


def run(plot_traj=True, calc_para=True, simulation=False, max_time=None, shape=None):
    # initialization
    path = 'gradU.txt'
    delta_t, list_A = read_grad(path)
    list_time = [delta_t * i for i in range(len(list_A))]
    solver = ParaSolver(list_A, list_time)

    if plot_traj:
        # position
        filename = 'Utr.txt'
        delta_t, list_position = get_position(filename)
        plot_position(list_position, delta=delta, max_time=max_time, shape=shape)

    if simulation:
        # visualization
        list_eig_values, list_eig_vectors = solver.calc_eig_para()
        list_length = np.sqrt(1 / (np.array(list_eig_values) + 1e-8))
        simulation_3d(list_length, list_eig_vectors)

    # data
    if calc_para:
        solver.calc_geo_para(normalize=True)
        solver.calc_coli(normalize=True)
        max_time = 3
        solver.plot_ratio(log=False)
        solver.plot_angle()
        solver.plot_coli()

    # solver.calc_omega()
    # solver.calc_coli_omega()
    # solver.plot_coli_o()
