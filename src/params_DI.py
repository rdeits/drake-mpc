import numpy as np
import scipy.signal as sig

# discretization
t_s = .05
N_ocp = 2
N_mpc = 100

# system dynamics
A = np.array([[1., t_s], [0., 1.]])
B = np.array([[t_s**2], [t_s]])
[n_x, n_u] = np.shape(B)

# variable bounds
u_max = np.array([[1.]])
u_min = -u_max
x_max = np.array([[10.],[.5]])
x_min = -x_max

# OCP cost function
Q = np.array([[1., 0.], [0., 0.]])
R = np.array([[1.]])

# terminal constraints(available options: 'moas', 'none')
ter_cons = 'none'

# initial state
x0 = np.array([[1.09],[0.]])

# test point for explicit mpc
x_test = np.array([[3.],[0.]])