from exp.exp_sing.experiment import run


##############################################################
###############     EDIT PARAMETERS HERE   ###################

# plot trajectory ?
plot_traj = True

# want to note points ?
max_time = None

# compute deformation ?
calc_para = True

# output the animation ?
simulation = False

# shape ?
shape = None

###############              END            ##################
##############################################################

if __name__ == '__main__':
    run(plot_traj=plot_traj, calc_para=calc_para, simulation=simulation, max_time=max_time, shape=shape)
