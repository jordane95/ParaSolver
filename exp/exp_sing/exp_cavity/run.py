from common.fileIO import read_grad, get_position
from common.paraSolver import ParaSolver
from common.visualization import simulation_3d, plot_position


def run():
    path = 'gradU-0.001.txt'
    delta_t, list_A = read_grad(path)
    list_time = [delta_t*i for i in range(len(list_A))]
    solver = ParaSolver(list_A, list_time)

    # # plot trajectory
    filename = 'Utr-0.001.txt'
    delta_t, list_position = get_position(filename)
    plot_position(list_position, delta=0.001, max_time=1.6)

    # # visualization
    # list_eig_values, list_eig_vectors = solver.calc_eig_para()
    # list_length = np.sqrt(1/(np.array(list_eig_values)+1e-8))
    # print('calculation done')
    # simulation_3d(list_length, list_eig_vectors)

    # # data for analyse
    # solver.calc_geo_para(normalize=True)
    # solver.calc_coli(normalize=True)
    #
    # solver.plot_ratio()
    # solver.plot_angle()
    # solver.plot_coli()


if __name__ == '__main__':
    run()
