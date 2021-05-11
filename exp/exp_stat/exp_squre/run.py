from exp.exp_stat.experiment import run

######################################################
############  CUSTOMIZE PARAMETERS HERE  #############

# number of particles to track
nums = 3

# time_step (i.e. first line of the grad.txt file)
delta_t = 0.0008

# how many steps ( < number lines of grad.txt file)
steps = 480

# calculate the deformation parameters or not
calc_para = True

# plot the trajectory in the screen or not
plot_traj = False

# save the trajectory in the desk "./img/" or not
save_traj = True

# shape of the fluid flow, optional: ['l']
shape = None

#############           END          ################
#####################################################

if __name__ == '__main__':
    run(nums=nums, delta_t=delta_t, steps=steps, calc_para=calc_para, plot_traj=plot_traj, save_traj=save_traj, shape=shape)
