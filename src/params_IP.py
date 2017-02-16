import numpy as np
import scipy.signal as sig

# dynamics
m = 1.
l = 1.
g = 10.

# discretization
t_s = .1
N_ocp = 3
N_mpc = 50

# system dynamics
A = np.array([[0., 1.], [g/l, 0.]])
B = np.array([[0.], [1/(m*l**2.)]])
[n_x, n_u] = np.shape(B)
C = np.eye(n_x)
n_y = np.shape(C)[0]
D = np.zeros((n_y,n_u))
sys = sig.cont2discrete((A,B,C,D),t_s,'zoh')
[A, B] = sys[0:2]

# variable bounds
x_max = np.array([[np.pi/6.],[np.pi]])
x_min = -x_max
u_max = np.array([x_max[0]*m*g*l*3./4.])
# u_max = np.array([[5.]])
u_min = -u_max
# OCP cost function
Q = np.eye(n_x)/100.
R = np.eye(n_u)

# terminal constraints(available options: 'moas', 'none')
ter_cons = 'moas'

# initial state
x0 = np.array([[1.],[0.]])
x0 = np.array([[.3],[0.]])

# test point for explicit mpc
x_test = np.array([[-1.09],[1.]])
x_test = np.array([[0.],[0.]])