from common.fileIO import read_grad, read_position
from common.paraSolver import ParaSolver
from common.visualization import simulation_3d, plot_position


def run():
    # initialization
    path = 'gradU.txt'
    delta_t, list_A = read_grad(path)
    list_time = [delta_t * i for i in range(len(list_A))]
    solver = ParaSolver(list_A, list_time)

    # position
    position_file = 'Utr.txt'
    delta, list_position, list_velocity = read_position(position_file)
    plot_position(list_position, delta=0.01, max_time=3)

    # simulation
    # list_eig_values, list_eig_vectors = solver.calc_eig_para(sort=True)
    # list_length = np.sqrt(1 / (np.array(list_eig_values) + 1e-8))
    # simulation_3d(list_length, list_eig_vectors)

    # data
    # solver.calc_geo_para(normalize=True)
    # solver.calc_coli(normalize=True)
    #
    # solver.plot_ratio(log=False)
    # solver.plot_angle()
    # solver.plot_coli()


if __name__ == '__main__':
    run()
