from mpc_functions import *

def plot_input_seq(u_seq, u_min, u_max, t_s, N):
    """
    plot_input_seq(u_seq, u_min, u_max, t_s, N):
    plots the input sequences as functions of time
    INPUTS:
    u_seq -> input sequence \in R^(N*n_u)
    [u_min, u_max]  -> lower and upper bound on the input
    t_s   -> sampling time
    N     -> time steps
    """
    n_u = u_seq.shape[0]/N
    u_seq = np.reshape(u_seq,(n_u,N), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_u):
        plt.subplot(n_u, 1, i+1)
        inPlot, = plt.step(t, np.hstack((u_seq[i,0],u_seq[i,:])),'b')
        x_lbPlot, = plt.step(t, u_min[i,0]*np.ones(t.shape),'r')
        plt.step(t, u_max[i,0]*np.ones(t.shape),'r')
        plt.ylabel(r'$u_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            plt.legend([inPlot, x_lbPlot], ['Optimal control', 'Control bounds'], loc=1)
    plt.xlabel(r'$t$')

def plot_state_traj(x_traj, x_min, x_max, t_s, N):
    """
    plot_state_traj(x_traj, x_min, x_max, t_s, N):
    plots the state trajectories as functions of time
    INPUTS:
    x_traj          -> state trajectory \in R^((N+1)*n_x)
    [x_min, x_max]  -> lower and upper bound on the state
    t_s             -> sampling time
    N               -> time steps
    """
    n_x = x_traj.shape[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    t = np.linspace(0,N*t_s,N+1)
    for i in range(0, n_x):
        plt.subplot(n_x, 1, i+1)
        stPlot, = plt.plot(t, x_traj[i,:],'b')
        x_lbPlot, = plt.step(t, x_min[i,0]*np.ones(t.shape),'r')
        plt.step(t, x_max[i,0]*np.ones(t.shape),'r')
        plt.ylabel(r'$x_{' + str(i+1) + '}$')
        plt.xlim((0.,N*t_s))
        if i == 0:
            plt.legend([stPlot, x_lbPlot], ['Optimal trajectory', 'State bounds'], loc=1)
    plt.xlabel(r'$t$')

def plot_state_space_traj(x_traj, N, col='b'):
    """
    plot_state_space_traj(x_traj, N, col='b'):
    plots the state trajectories as functions of time (ONLY 2D)
    INPUTS:
    x_traj -> state trajectory \in R^((N+1)*n_x)
    N      -> time steps
    col    -> line specs
    OUTPUTS:
    traj_plot -> figure handle
    """
    n_x = x_traj.shape[0]/(N+1)
    x_traj = np.reshape(x_traj,(n_x,N+1), 'F')
    plt.scatter(x_traj[0,0], x_traj[1,0], color=col, alpha=.5)
    plt.scatter(x_traj[0,-1], x_traj[1,-1], color=col, marker='s', alpha=.5)
    traj_plot = plt.plot(x_traj[0,:], x_traj[1,:], color=col)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    return traj_plot

def plot_cons_moas(A, lhs_x, rhs_x, t):
    """
    plot_cons_moas(A, lhs_x, rhs_x, t):
    plots the set of constraints that define the MOAS until the first redundant polytope
    INPUTS:
    A          -> state transition matrix
    [lhs_x, rhs_x] -> state constraint lhs_x x(k) <= rhs_x \forall k
    t          -> number of steps after that the constraints are redundant
    OUTPUTS:
    [active_plot, redund_plot] -> figure handles
    """
    # plot constraints from the first redundant one
    for i in range(t+1,-1,-1):
        lhs_x_i = lhs_x.dot(np.linalg.matrix_power(A,i))
        print 'start'
        print rhs_x
        print 'stop'
        poly_x_i = Poly(lhs_x_i, rhs_x*1.) # ? if I don't do this, the modification to this value that I make inside the class changes also the value otuside the class!!!!
        if i == t+1:
            redund_plot = poly_x_i.plot2d('g-.')
        else:
            active_plot = poly_x_i.plot2d('y-.')
    return [active_plot, redund_plot]

def plot_moas(lhs_moas, rhs_moas, t, A, N=0):
    """
    plot_moas(lhs_moas, rhs_moas, t, A, N):
    plots the maximum output admissible set and a trajectory for each vertex of the moas
    INPUTS:
    [lhs_moas, rhs_moas] -> matrices such that moas := {x | lhs_moas*x <= rhs_moas}
    t                  -> number of steps after that the constraints are redundant
    A                  -> state transition matrix
    N                  -> number of steps for the simulations
    OUTPUTS:
    [moas_plot, traj_plot] -> figure handles
    """
    n_x = A.shape[0]
    # plot MOAS polyhedron
    poly_moas = Poly(lhs_moas, rhs_moas)
    moas_plot = poly_moas.plot2d('r-')
    # simulate a trajectory for each vertex
    for i in range(0, poly_moas.verts.shape[0]):
        vert = poly_moas.verts[i,:].reshape(n_x,1)
        x_traj = sim_lin_sys(vert, N, A)
        traj_plot, = plot_state_space_traj(x_traj,N)
    return [moas_plot, traj_plot]

def plot_nom_traj(x_traj_qp, x_traj_lqr, lhs_moas, rhs_moas):
    """
    plot_nom_traj(x_traj_qp, lhs_moas, rhs_moas):
    plots the open-loop optimal trajectories for each sampling time (ONLY 2D)
    INPUTS:
    x_traj_qp            -> matrix with optimal trajectories
    x_traj_lqr           -> matrix with optimal states
    [lhs_moas, rhs_moas] -> matrices such that moas := {x | lhs_moas*x <= rhs_moas}
    """
    n_traj = np.shape(x_traj_qp)[1]
    N_ocp = (np.shape(x_traj_qp)[0]-1)/2
    col_map = plt.get_cmap('jet')
    col_norm  = mpl.colors.Normalize(vmin=0, vmax=n_traj)
    scalar_map = mpl.cm.ScalarMappable(norm=col_norm, cmap=col_map)
    poly_moas = Poly(lhs_moas, rhs_moas)
    moas_plot = poly_moas.plot2d('r-')
    leg_plot = [moas_plot]
    leg_lab = ['MOAS']
    for i in range(0,n_traj):
        col = scalar_map.to_rgba(i)
        leg_plot += plot_state_space_traj(x_traj_qp[:,i], N_ocp, col)
        leg_lab += [r'$\mathbf{x}^*(x_{' + str(i) + '})$']
    for i in range(0,np.shape(x_traj_lqr)[1]):
        if i == 0:
            leg_plot += [plt.scatter(x_traj_lqr[0,i], x_traj_lqr[1,i], color='b', marker='d', alpha=.5)]
            leg_lab += [r'LQR']
        else:
            plt.scatter(x_traj_lqr[0,i], x_traj_lqr[1,i], color='b', marker='d', alpha=.5)
    plt.legend(leg_plot, leg_lab, loc=1)
