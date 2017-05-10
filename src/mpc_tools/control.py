import time
import sys, os
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import gurobipy as grb
from contextlib import contextmanager
from optimization.pnnls import linear_program
from optimization.gurobi import quadratic_program
from geometry import Polytope
from dynamical_systems import DTAffineSystem, DTPWASystem
from mpcqp import CanonicalMPCQP
from optimization.mpqpsolver import MPQPSolver



class MPCController:

    def __init__(self, sys, N, objective_norm, Q, R, P=None, X=None, U=None, X_N=None):
        self.sys = sys
        self.N = N
        self.objective_norm = objective_norm
        self.Q = Q
        self.R = R
        if P is None:
            self.P = Q
        else:
            self.P = P
        self.X = X
        self.U = U
        if X_N is None and X is not None:
            self.X_N = X
        else:
            self.X_N = X_N
        self.condense_program()
        return

    def condense_program(self):
        c = np.zeros((self.sys.n_x, 1))
        a_sys = DTAffineSystem(self.sys.A, self.sys.B, c)
        sys_list = [a_sys]*self.N
        X_list = [self.X]*self.N
        U_list = [self.U]*self.N
        switching_sequence = [0]*self.N
        pwa_sys = DTPWASystem(sys_list, X_list, U_list)
        self.condensed_program = OCP_condenser(pwa_sys, self.objective_norm, self.Q, self.R, self.P, self.X_N, switching_sequence)
        self.remove_intial_state_contraints()
        return

    def remove_intial_state_contraints(self, tol=1e-10):
        C_u_rows_norm = list(np.linalg.norm(self.condensed_program.C_u, axis=1))
        intial_state_contraints = [i for i, row_norm in enumerate(C_u_rows_norm) if row_norm < tol]
        if len(intial_state_contraints) > self.X.lhs_min.shape[0]:
            raise ValueError('Wrong number of zero rows in the constrinats')
        self.condensed_program.C_u = np.delete(self.condensed_program.C_u,intial_state_contraints, 0)
        self.condensed_program.C_x = np.delete(self.condensed_program.C_x,intial_state_contraints, 0)
        self.condensed_program.C = np.delete(self.condensed_program.C,intial_state_contraints, 0)
        return

    def feedforward(self, x0):
        u_feedforward, cost = self.condensed_program.solve(x0)
        if any(np.isnan(u_feedforward).flatten()):
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
        return u_feedforward, cost

    def feedback(self, x0):
        u_feedforward = self.feedforward(x0)[0]
        u_feedback = u_feedforward[0:self.sys.n_u]
        return u_feedback

class MPCExplicitController:

    def __init__(self, mpqp):
        mpqp.remove_linear_terms()
        self.mpqp = mpqp
        mpqp_solution = MPQPSolver(mpqp)
        self.critical_regions = mpqp_solution.critical_regions
        return

    def feedforward(self, x0):
        cr_x0 = self.critical_regions.lookup(x0)
        if cr_x0 is not None:
            u_feedforward = cr_x0.u_offset + cr_x0.u_linear.dot(x0)
            cost = .5*x0.T.dot(cr_x0.V_quadratic).dot(x0) + cr_x0.V_linear.dot(x0) + cr_x0.V_offset
            cost = cost[0,0]
        else:
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = np.full((self.mpqp.G.shape[1], 1), np.nan)
            cost = np.nan
        return u_feedforward, cost

    def feedback(self, x0):
        return self.feedforward(x0)[0:self.sys.n_u]

    def optimal_value_function(self, x0):
        cr_x0 = self.critical_regions.lookup(x0)
        if cr_x0 is not None:
            cost = .5*x0.T.dot(cr_x0.V_quadratic).dot(x0) + cr_x0.V_linear.dot(x0) + cr_x0.V_offset
        else:
            #print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            cost = np.nan
        return cost

    def group_critical_regions(self):
        self.u_offset_list = []
        self.u_linear_list = []
        self.cr_families = []
        for cr in self.critical_regions:
            cr_family = np.where(np.isclose(cr.u_offset[0], self.u_offset_list))[0]
            if cr_family and all(np.isclose(cr.u_linear[0,:], self.u_linear_list[cr_family[0]])):
                self.cr_families[cr_family[0]].append(cr)
            else:
                self.cr_families.append([cr])
                self.u_offset_list.append(cr.u_offset[0])
                self.u_linear_list.append(cr.u_linear[0,:])
        print 'Critical regions grouped in ', str(len(self.cr_families)), ' sets.'
        return



class MPCHybridController:

    def __init__(self, sys, N, objective_norm, Q, R, P, X_N):
        self.sys = sys
        self.N = N
        self.objective_norm = objective_norm
        self.Q = Q
        self.R = R
        self.P = P
        self.X_N = X_N
        self.compute_big_M_domains()
        self.compute_big_M_dynamics()
        return

    def compute_big_M_domains(self):
        self.big_M_domains = []
        for i in range(self.sys.n_sys):
            big_M_i = []
            lhs_i = linalg.block_diag(self.sys.state_domains[i].lhs_min, self.sys.input_domains[i].lhs_min)
            rhs_i = np.vstack((self.sys.state_domains[i].rhs_min, self.sys.input_domains[i].rhs_min))
            for j in range(self.sys.n_sys):
                big_M_ij = []
                if i != j:
                    lhs_j = linalg.block_diag(self.sys.state_domains[j].lhs_min, self.sys.input_domains[j].lhs_min)
                    rhs_j = np.vstack((self.sys.state_domains[j].rhs_min, self.sys.input_domains[j].rhs_min))
                    for k in range(lhs_i.shape[0]):
                        big_M_ijk = - linear_program(-lhs_i[k,:], lhs_j, rhs_j)[1] - rhs_i[k]
                        big_M_ij.append(big_M_ijk[0])
                big_M_ij = np.reshape(big_M_ij, (len(big_M_ij), 1))
                big_M_i.append(big_M_ij)
            self.big_M_domains.append(big_M_i)
        return

    def compute_big_M_dynamics(self):
        self.big_M_dynamics = []
        self.small_m_dynamics = []
        for i in range(self.sys.n_sys):
            big_M_i = []
            small_m_i = []
            lhs_i = np.hstack((self.sys.affine_systems[i].A, self.sys.affine_systems[i].B))
            rhs_i = self.sys.affine_systems[i].c
            for j in range(self.sys.n_sys):
                big_M_ij = []
                small_m_ij = []
                lhs_j = linalg.block_diag(self.sys.state_domains[j].lhs_min, self.sys.input_domains[j].lhs_min)
                rhs_j = np.vstack((self.sys.state_domains[j].rhs_min, self.sys.input_domains[j].rhs_min))
                for k in range(lhs_i.shape[0]):
                    big_M_ijk = - linear_program(-lhs_i[k,:], lhs_j, rhs_j)[1] + rhs_i[k]
                    big_M_ij.append(big_M_ijk[0])
                    small_m_ijk = linear_program(lhs_i[k,:], lhs_j, rhs_j)[1] + rhs_i[k]
                    small_m_ij.append(small_m_ijk[0])
                big_M_ij = np.reshape(big_M_ij, (len(big_M_ij), 1))
                small_m_ij = np.reshape(small_m_ij, (len(small_m_ij), 1))
                big_M_i.append(big_M_ij)
                small_m_i.append(np.array(small_m_ij))
            self.big_M_dynamics.append(big_M_i)
            self.small_m_dynamics.append(small_m_i)
        return

    def feedforward(self, x0):

        # gurobi model
        model = grb.Model()

        # variables
        x, model = grb_real_var(model, [self.N+1, self.sys.n_x])
        u, model = grb_real_var(model, [self.N, self.sys.n_u])
        z, model = grb_real_var(model, [self.N, self.sys.n_sys, self.sys.n_x])
        d = model.addVars(self.N, self.sys.n_sys, vtype=grb.GRB.BINARY, name='d')
        model.update()

        # numpy variables (list of numpy matrices ordered in time)
        x_np = [np.array([[x[k,i]] for i in range(self.sys.n_x)]) for k in range(self.N+1)]
        u_np = [np.array([[u[k,i]] for i in range(self.sys.n_u)]) for k in range(self.N)]

        # set objective
        model = self.mip_objective(model, x_np, u_np)

        # initial condition
        model.addConstrs((x[0,i] == x0[i,0] for i in range(self.sys.n_x)))

        # set constraints
        model = self.mip_constraints(model, x_np, u_np, z, d)

        # run optimization
        model.setParam('OutputFlag', False)
        model.optimize()

        # return solution
        if model.status != grb.GRB.Status.OPTIMAL:
            print('Unfeasible initial condition x_0 = ' + str(x0.tolist()))
            u_feedforward = np.full((self.sys.n_u*self.N,1), np.nan)
            cost = np.nan
            switching_sequence = [np.nan]*self.N
        else:
            cost = model.objVal
            u_feedforward = np.array([[model.getAttr('x', u)[k,i] for i in range(self.sys.n_u)] for k in range(self.N)])
            d_star = [np.array([[model.getAttr('x', d)[k,i]] for i in range(self.sys.n_sys)]) for k in range(self.N)]
            switching_sequence = [np.where(np.isclose(d, 1.))[0][0] for d in d_star]
        return u_feedforward, cost, switching_sequence

    def mip_objective(self, model, x_np, u_np):

        # linear objective
        if self.objective_norm == 'one':
            phi = model.addVars(self.N+1, self.sys.n_x, name='phi')
            psi = model.addVars(self.N, self.sys.n_u, name='psi')
            model.update()
            V = 0.
            for k in range(self.N+1):
                for i in range(self.sys.n_x):
                    V += phi[k,i]
            for k in range(self.N):
                for i in range(self.sys.n_u):
                    V += psi[k,i]
            model.setObjective(V)
            for k in range(self.N):
                for i in range(self.sys.n_x):
                    model.addConstr(phi[k,i] >= self.Q[i,:].dot(x_np[k])[0])
                    model.addConstr(phi[k,i] >= - self.Q[i,:].dot(x_np[k])[0])
                for i in range(self.sys.n_u):
                    model.addConstr(psi[k,i] >= self.R[i,:].dot(u_np[k])[0])
                    model.addConstr(psi[k,i] >= - self.R[i,:].dot(u_np[k])[0])
            for i in range(self.sys.n_x):
                model.addConstr(phi[self.N,i] >= self.P[i,:].dot(x_np[self.N])[0])
                model.addConstr(phi[self.N,i] >= - self.P[i,:].dot(x_np[self.N])[0])

       # quadratic objective 
        elif self.objective_norm == 'two':
            V = 0.
            for k in range(self.N):
                V += x_np[k].T.dot(self.Q).dot(x_np[k]) + u_np[k].T.dot(self.R).dot(u_np[k])
            V += x_np[self.N].T.dot(self.P).dot(x_np[self.N])
            model.setObjective(V[0,0])

        return model

    def mip_constraints(self, model, x_np, u_np, z, d):

        with suppress_stdout():

            # disjuction
            for k in range(self.N):
                model.addConstr(np.sum([d[k,i] for i in range(self.sys.n_sys)]) == 1.)

            # relaxation of the domains
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr_x = self.sys.state_domains[i].lhs_min.dot(x_np[k]) - self.sys.state_domains[i].rhs_min
                    expr_u = self.sys.input_domains[i].lhs_min.dot(u_np[k]) - self.sys.input_domains[i].rhs_min
                    expr_xu = np.vstack((expr_x, expr_u))
                    expr_big_M = np.sum([self.big_M_domains[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    expr = expr_xu - expr_big_M
                    model.addConstrs((expr[j][0] <= 0. for j in range(len(expr))))

            # state transition
            for k in range(self.N):
                for j in range(self.sys.n_x):
                    expr = 0.
                    for i in range(self.sys.n_sys):
                        expr += z[k,i,j]
                    model.addConstr(x_np[k+1][j,0] == expr)

            # relaxation of the dynamics, part 1
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr_big_M = self.big_M_dynamics[i][i]*d[k,i]
                    expr_small_m = self.small_m_dynamics[i][i]*d[k,i]
                    for j in range(self.sys.n_x):
                        model.addConstr(z[k,i,j] <= expr_big_M[j,0])
                        model.addConstr(z[k,i,j] >= expr_small_m[j,0])
            
            # relaxation of the dynamics, part 2
            for k in range(self.N):
                for i in range(self.sys.n_sys):
                    expr = self.sys.affine_systems[i].A.dot(x_np[k]) + self.sys.affine_systems[i].B.dot(u_np[k]) + self.sys.affine_systems[i].c
                    expr_big_M = expr - np.sum([self.big_M_dynamics[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    expr_small_m = expr - np.sum([self.small_m_dynamics[i][j]*d[k,j] for j in range(self.sys.n_sys) if j != i], axis=0)
                    for j in range(self.sys.n_x):
                        model.addConstr(z[k,i,j] >= expr_big_M[j,0])
                        model.addConstr(z[k,i,j] <= expr_small_m[j,0])

            # terminal constraint
            expr = self.X_N.lhs_min.dot(x_np[self.N]) - self.X_N.rhs_min
            model.addConstrs((expr[i,0] <= 0. for i in range(len(self.X_N.minimal_facets))))

        return model

    def feedback(self, x0):
        return self.feedforward(x0)[0][0:self.sys.n_u]

    def condense_program(self, switching_sequence):
        if len(switching_sequence) != self.N:
            raise ValueError('Switching sequence not coherent with the controller horizon.')
        return OCP_condenser(self.sys, self.objective_norm, self.Q, self.R, self.P, self.X_N, switching_sequence)

    def backward_reachability_analysis(self, switching_sequence):
        if self.X_N is None:
            raise ValueError('A terminal constraint is needed for the backward reachability analysis!')
        if len(switching_sequence) != self.N:
            raise ValueError('Switching sequence not coherent with the controller horizon.')
        print('Computing feasible set for the switching sequence ' + str(switching_sequence))
        tic = time.time()
        feasible_set = self.X_N
        A_sequence = [self.sys.affine_systems[switch].A for switch in switching_sequence]
        B_sequence = [self.sys.affine_systems[switch].B for switch in switching_sequence]
        c_sequence = [self.sys.affine_systems[switch].c for switch in switching_sequence]
        U_sequence = [self.sys.input_domains[switch] for switch in switching_sequence]
        X_sequence = [self.sys.state_domains[switch] for switch in switching_sequence]
        for i in range(self.N-1,-1,-1):
            lhs_x = feasible_set.lhs_min.dot(A_sequence[i])
            lhs_u = feasible_set.lhs_min.dot(B_sequence[i])
            lhs = np.hstack((lhs_x, lhs_u))
            rhs = feasible_set.rhs_min - feasible_set.lhs_min.dot(c_sequence[i])
            feasible_set = Polytope(lhs, rhs)
            lhs = linalg.block_diag(X_sequence[i].lhs_min, U_sequence[i].lhs_min)
            rhs = np.vstack((X_sequence[i].rhs_min, U_sequence[i].rhs_min))
            feasible_set.add_facets(lhs, rhs)
            feasible_set.assemble()
            feasible_set = feasible_set.orthogonal_projection(range(self.sys.n_x))
        toc = time.time()
        print('Feasible set computed in ' + str(toc-tic) + ' s')
        return feasible_set

    def plot_feasible_set(self, switching_sequence, **kwargs):
        feasible_set = self.backward_reachability_analysis(switching_sequence)
        feasible_set.plot(**kwargs)
        plt.text(feasible_set.center[0], feasible_set.center[1], str(switching_sequence))
        return

    # def bound_optimal_value_function(self, switching_sequence, X):

    #     prog = self.condense_program(switching_sequence)

    #     # lower bound
    #     H = np.vstack((
    #         np.hstack((prog.F_xx, prog.F_xu)),
    #         np.hstack((prog.F_xu.T, prog.F_uu))
    #         ))
    #     f = np.vstack((prog.F_x, prog.F_u))
    #     A = np.vstack((
    #         np.hstack((- prog.C_x, prog.C_u)),
    #         np.hstack((X.lhs_min, np.zeros((X.lhs_min.shape[0], prog.C_u.shape[1]))))
    #         ))
    #     b = np.vstack((prog.C, X.rhs_min))
    #     x_min, lb = quadratic_program(H, f, A, b)
    #     #x_min = x_min[0:self.sys.n_x,:]
    #     lb += prog.F[0,0]

    #     # upper bound
    #     cost_vertices = []
    #     for vertex in X.vertices:
    #         u_star, cost = prog.solve(vertex)
    #         cost_vertices.append(cost)
    #         # tol = 1.e-5
    #         # residuals = np.abs(prog.C_u.dot(u_star) - prog.C_x.dot(vertex) - prog.C).flatten()
    #         # active_set = np.where(residuals > tol)[0]
    #     ub = max(cost_vertices)
    #     return lb, ub






    # def plot_feasible_set(self, switching_sequence, **kwargs):
    #     tic = time.time()
    #     mpqp = self.condense_qp(switching_sequence, self.sys, self.Q, self.R, self.P, self.X_N)
    #     if mpqp is not None:
    #         X = self.sys.X_list[switching_sequence[0]]
    #         lhs = np.vstack((mpqp.feasible_set.lhs_min, X.lhs_min))
    #         rhs = np.vstack((mpqp.feasible_set.rhs_min, X.rhs_min))
    #         p = Polytope(lhs,rhs)
    #         p.assemble()
    #         toc = time.time()
    #         print('Feasible set computed in ' + str(toc-tic) + ' s')
    #         p.plot(**kwargs)
    #         plt.text(p.center[0], p.center[1], str(switching_sequence))
    #     else:
    #         print('Unfeasible switching sequence!')
    #     return




class parametric_lp:

    def __init__(self, F_u, F_x, F, C_u, C_x, C):
        """
        LP in the form:
        min  \sum_i | (F_u u + F_x x + F)_i |
        s.t. C_u u <= C_x x + C
        """
        self.F_u = F_u
        self.F_x = F_x
        self.F = F
        self.C_u = C_u
        self.C_x = C_x
        self.C = C
        self.add_slack_variables()
        return

    def add_slack_variables(self):
        """
        Reformulates the LP as:
        min f^T z
        s.t. A z <= B x + c
        """
        n_slack = self.F.shape[0]
        n_u = self.F_u.shape[1]
        self.f = np.vstack((
            np.zeros((n_u,1)),
            np.ones((n_slack,1))
            ))
        self.A = np.vstack((
            np.hstack((self.C_u, np.zeros((self.C_u.shape[0], n_slack)))),
            np.hstack((self.F_u, -np.eye(n_slack))),
            np.hstack((-self.F_u, -np.eye(n_slack)))
            ))
        self.B = np.vstack((self.C_x, -self.F_x, self.F_x))
        self.c = np.vstack((self.C, -self.F, self.F))
        self.n_var = n_u + n_slack
        self.n_cons = self.A.shape[0]
        return

    def solve(self, x0):
        x0 = np.reshape(x0, (x0.shape[0], 1))
        u_star, V_star =  linear_program(self.f, self.A, self.B.dot(x0)+self.c)
        u_star = u_star[0:self.F_u.shape[1]]
        return u_star, V_star


class parametric_qp:

    def __init__(self, F_uu, F_xu, F_xx, F_u, F_x, F, C_u, C_x, C):
        """
        Multiparametric QP in the form:
        min  .5 u' F_{uu} u + x0' F_{xu} u + F_u' u + .5 x0' F_{xx} x0 + F_x' x0 + F
        s.t. C_u u <= C_x x + C
        """
        self.F_uu = F_uu
        self.F_xx = F_xx
        self.F_xu = F_xu
        self.F_u = F_u
        self.F_x = F_x
        self.F = F
        self.C_u = C_u
        self.C_x = C_x
        self.C = C
        self._feasible_set = None
        return

    def solve(self, x0):
        x0 = np.reshape(x0, (x0.shape[0], 1))
        H = self.F_uu
        f = x0.T.dot(self.F_xu) + self.F_u.T
        A = self.C_u
        b = self.C + self.C_x.dot(x0)
        u_star, cost = quadratic_program(H, f, A, b)
        cost += .5*x0.T.dot(self.F_xx).dot(x0) + self.F_x.T.dot(x0) + self.F
        return u_star, cost[0,0]

    def remove_linear_terms(self):
        """
        Applies the change of variables z = u + F_uu^-1 (F_xu' x + F_u')
        """
        self.H_inv = np.linalg.inv(self.F_uu)
        self.H = self.F_uu
        self.F_xx_q = (self.F_xx - self.F_xu.dot(self.H_inv).dot(self.F_xu.T))
        self.F_x_q = self.F_x - self.F_u.T.dot(self.H_inv).dot(self.F_xu.T)
        self.F_q = self.F - .5*self.F_u.T.dot(self.H_inv).dot(self.
            F_u)
        self.G = self.C_u
        self.S = self.C_x + self.C_u.dot(self.H_inv).dot(self.F_xu.T)
        self.W = self.C + self.C_u.dot(self.H_inv).dot(self.F_u)
        return

    @property
    def feasible_set(self):
        if self._feasible_set is None:
            augmented_polytope = Polytope(np.hstack((- self.C_x, self.C_u)), self.C)
            augmented_polytope.assemble()
            self._feasible_set = augmented_polytope.orthogonal_projection(range(self.C_x.shape[1]))
        return self._feasible_set



### AUXILIARY FUNCTIONS

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def OCP_condenser(sys, objective_norm, Q, R, P, X_N, switching_sequence):
    N = len(switching_sequence)
    Q_bar = linalg.block_diag(*[Q for i in range(N)] + [P])
    R_bar = linalg.block_diag(*[R for i in range(N)])
    G, W, E = constraint_condenser(sys, X_N, switching_sequence)
    if objective_norm == 'one':
        F_u, F_x, F = linear_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = parametric_lp(F_u, F_x, F, G, E, W)
    elif objective_norm == 'two':
        F_uu, F_xu, F_xx, F_u, F_x, F = quadratic_objective_condenser(sys, Q_bar, R_bar, switching_sequence)
        parametric_program = parametric_qp(F_uu, F_xu, F_xx, F_u, F_x, F, G, E, W)
    return parametric_program

def constraint_condenser(sys, X_N, switching_sequence):
    N = len(switching_sequence)
    G_u, W_u, E_u = input_constraint_condenser(sys, switching_sequence)
    G_x, W_x, E_x = state_constraint_condenser(sys, X_N, switching_sequence)
    G = np.vstack((G_u, G_x))
    W = np.vstack((W_u, W_x))
    E = np.vstack((E_u, E_x))
    p = Polytope(np.hstack((G, -E)), W)
    p.assemble()
    if not p.empty:
        G = p.lhs_min[:,:sys.n_u*N]
        E = - p.lhs_min[:,sys.n_u*N:]
        W = p.rhs_min
    else:
        G = None
        W = None
        E = None
    return G, W, E

def input_constraint_condenser(sys, switching_sequence):
    N = len(switching_sequence)
    U_sequence = [sys.input_domains[switching_sequence[i]] for i in range(N)]
    G_u = linalg.block_diag(*[U.lhs_min for U in U_sequence])
    W_u = np.vstack([U.rhs_min for U in U_sequence])
    E_u = np.zeros((W_u.shape[0], sys.n_x))
    return G_u, W_u, E_u

def state_constraint_condenser(sys, X_N, switching_sequence):
    N = len(switching_sequence)
    X_sequence = [sys.state_domains[switching_sequence[i]] for i in range(N)]
    lhs_x = linalg.block_diag(*[X.lhs_min for X in X_sequence] + [X_N.lhs_min])
    rhs_x = np.vstack([X.rhs_min for X in X_sequence] + [X_N.rhs_min])
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    G_x = lhs_x.dot(B_bar)
    W_x = rhs_x - lhs_x.dot(c_bar)
    E_x = - lhs_x.dot(A_bar)
    return G_x, W_x, E_x

def linear_objective_condenser(sys, Q_bar, R_bar, switching_sequence):
    """
    \sum_i | (F_u u + F_x x + F)_i |
    """
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    F_u = np.vstack((Q_bar.dot(B_bar), R_bar))
    F_x = np.vstack((Q_bar.dot(A_bar), np.zeros((R_bar.shape[0], A_bar.shape[1]))))
    F = np.vstack((Q_bar.dot(c_bar), np.zeros((R_bar.shape[0], 1))))
    return F_u, F_x, F

def quadratic_objective_condenser(sys, Q_bar, R_bar, switching_sequence):
    """
    .5 u' F_{uu} u + x0' F_{xu} u + F_u' u + .5 x0' F_{xx} x0 + F_x' x0 + F
    """
    A_bar, B_bar, c_bar = sys.condense(switching_sequence)
    F_uu = 2*(R_bar + B_bar.T.dot(Q_bar).dot(B_bar))
    F_xu = 2*A_bar.T.dot(Q_bar).dot(B_bar)
    F_xx = 2.*A_bar.T.dot(Q_bar).dot(A_bar)
    F_u = 2.*B_bar.T.dot(Q_bar).dot(c_bar)
    F_x = 2.*A_bar.T.dot(Q_bar).dot(c_bar)
    F = c_bar.T.dot(Q_bar).dot(c_bar)
    return F_uu, F_xu, F_xx, F_u, F_x, F

def grb_real_var(model, dimensions):
    lb_x = [-grb.GRB.INFINITY]
    for dimension in dimensions:
        lb_x = [lb_x * dimension]
    x = model.addVars(*dimensions, lb=lb_x, name='x')
    return x, model
