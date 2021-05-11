import matplotlib.pyplot as plt
import numpy as np


def _plot_ellipse_3d(lengths, trans_mat, ax):
    (a, b, c) = tuple(lengths)
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2 * np.pi, 30)
    list_x = a * np.outer(np.sin(theta), np.sin(phi))
    list_y = b * np.outer(np.sin(theta), np.cos(phi))
    list_z = c * np.outer(np.cos(theta), np.ones_like(phi))
    list_cor_old = [[x, y, z] for (x, y, z) in zip(list_x, list_y, list_z)]
    list_cor_new = np.array([np.dot(trans_mat, cor) for cor in list_cor_old])
    plt.cla()
    # ax.plot_surface(list_cor_new[:, 0], list_cor_new[:, 1], list_cor_new[:, 2], color='b')
    ax.plot_wireframe(list_cor_new[:, 0], list_cor_new[:, 1], list_cor_new[:, 2], color='b')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.pause(0.05)


def simulation_3d(list_length, list_vectors):
    print("Plotting...")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.ion()
    time = 0
    for (lengths, trans_mat) in zip(list_length, list_vectors):
        print("At time %d" % time)
        _plot_ellipse_3d(lengths, trans_mat, ax)
        time += 1
    print("Finished!")
    plt.ioff()
    plt.show()


# plot the trajectory of a particle
def plot_position(list_position, save_dest=None, delta=None, max_time=None, shape=None):
    # plot trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(list_position[:, 0], list_position[:, 1], list_position[:, 2], c='b', s=0.1)
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    points_idx = []
    # optional: tracer
    if max_time:
        time = 0
        while time <= max_time:
            points_idx.append(int(time/delta))
            time += 0.5
        for idx, i in enumerate(points_idx):
            ax.scatter(list_position[i, 0], list_position[i, 1], list_position[i, 2], c='r', marker='^', s=10)
            ax.text(list_position[i, 0], list_position[i, 1], list_position[i, 2], 't='+str(idx*0.5)+"s", c='r', fontsize=7)
    # optional: plot surface shape
    if shape == 'l':
        plot_l_pipe(ax)
    # optional: save fig
    if save_dest is not None:
        plt.savefig(save_dest)
    else:
        plt.show()


def plot_l_pipe(ax, d=0.1):
    r = d/2
    t = np.linspace(0, 2*np.pi, 50)
    # first cylinder
    for i in np.linspace(0, 0.3, 10):
        end_points = [0+r*np.cos(t), 0+r*np.sin(t), np.array([i]*50)]
        ax.plot(end_points[0], end_points[1], end_points[2], color='g')
    # middle part
    start = np.array([.1, 0, .3])
    thetas = np.linspace(0, np.pi/2, 10)
    for theta in thetas:
        a = [-np.cos(theta), 0, np.sin(theta)]
        b = [0, 1, 0]
        center = start+d*np.array(a)
        X = [None]*3
        for i in range(3):
            X[i] = center[i]+r*np.cos(t)*a[i]+r*np.sin(t)*b[i]
        # print(X)
        ax.plot(X[0], X[1], X[2], color='g')
    # second cylinder
    for i in np.linspace(0.1, 0.6, 30):
        end_points = [np.array([i]*50), 0+r*np.cos(t), 0.4+r*np.sin(t)]
        ax.plot(end_points[0], end_points[1], end_points[2], color='g')
    ax.set_xlim(-0.1, 0.5)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.1, 0.5)
