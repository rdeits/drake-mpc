module act

using JuMP
using Gurobi
using DataStructures: OrderedDict
using Iterators: subsets
using ConvexHull
using Polyhedra

function ball_center(x_poly, G, W, S)
    # Bemporad et a. 2002, eq. (14)
    hr = SimpleHRepresentation(x_poly)
    T = hr.A
    Z = hr.b

    model = Model(solver=GurobiSolver(OutputFlag=0))
    @variables model begin
        epsilon
        z[1:size(G, 2)]
        x[1:size(S, 2)]
    end
    @objective model Max epsilon
    @constraint model [i=1:size(T, 1)] dot(T[i, :], x) + epsilon * norm(T[i, :]) <= Z[i]
    @constraint model G * z .- S * x .<= W
    solve(model)
    getvalue.(x)
end

function mpc_model(x_bounds)
    model = Model(solver=GurobiSolver(OutputFlag=0))
    @variables model begin
        z[1:2]
    end



    # Taking a model directly from Tondel et al. 2003, Example 1
    dt = 0.05
    H = [1.079 0.076; 0.076 1.073]
    F = [1.109 1.036; 1.573 1.517]
    G = [
        1 0;
        0 1;
        -1 0;
        0 -1;
        dt 0;
        dt dt;
        -dt 0;
        -dt -dt
    ]
    W = [1; 1; 1; 1; 0.5; 0.5; 0.5; 0.5]
    S = [
        1.0 1.4;
        0.9 1.3;
        -1.0 -1.4;
        -0.9 -1.3;
        0.1 -0.9;
        0.1 -0.9;
        -0.1 0.9;
        -0.1 0.9
    ]

    x0 = ball_center(x_bounds, G, W, S)
    @show x0
    @variable model x[i=1:2] == x0[i]

    @constraints model begin
        G * z .<= W + S * x
    end

    @objective model Min (z' * H * z)[1]

    @assert act.verify_form(model)
    model, z, x
end


function verify_form(model::Model)
    isempty(model.obj.aff.vars) &&
    isempty(model.sdpconstr) &&
    isempty(model.quadconstr) &&
    isempty(model.conicconstrDuals) &&
    all(c.lb == -Inf || c.ub == Inf for c in model.linconstr)
end

function get_objective_matrix(model::Model, vars, params)
    @assert verify_form(model)
    H = zeros(model.numCols, model.numCols)
    for i in 1:length(model.obj.qcoeffs)
        v1 = model.obj.qvars1[i]
        v2 = model.obj.qvars2[i]
        c = model.obj.qcoeffs[i]
        H[v1.col, v2.col] = c
    end
    for p in params
        @assert all((@view H[p.col, :]) .== 0)
        @assert all((@view H[:, p.col]) .== 0)
    end
    H = H[[v.col for v in vars], [v.col for v in vars]]
    H + H' - Diagonal(diag(H))
end

function get_linconstr_matrices(model, vars, params)
    @assert verify_form(model)
    A = full(JuMP.prepConstrMatrix(model))
    c, lb, ub = JuMP.prepProblemBounds(model)
    @assert all(c .== 0)

    G = A[:,[v.col for v in vars]]
    S = -A[:, [p.col for p in params]]
    W = zeros(size(A, 1))
    for i in 1:size(A, 1)
        if lb[i] == -Inf
            W[i] = ub[i]
        else
            @assert ub[i] == Inf
            W[i] = -lb[i]
            G[i,:] .= -G[i,:]
            S[i,:] .= -S[i,:]
        end
    end
    G, W, S
end

function linearly_independent_subset(G, Ai)
    for ai in subsets(Ai, size(G, 2))
        if rank(@view G[ai,:]) == size(G, 2)
            return ai
        end
    end
    error("Could not find a linearly independent subset")
end

function critical_region(model, vars, params)
    H = get_objective_matrix(model, vars, params)
    G, W, S = get_linconstr_matrices(model, vars, params)

    Ai = [i for (i, constr) in enumerate(model.linconstr) if isapprox(getvalue(constr.terms), constr.ub) || isapprox(getvalue(constr.terms), constr.lb)]
    if length(Ai) > size(G, 2)
        Ai = linearly_independent_subset(G, Ai)
    end
    GA = @view G[Ai,:]
    WA = @view W[Ai,:]
    SA = @view S[Ai,:]
    @show GA
    if rank(GA) < size(GA, 1)
        error("LICQ is violated, and we don't handle that yet")
    end

    crmodel = Model()
    @variable crmodel x[1:length(params)]
    lambdaA = vec(-inv(GA * inv(H) * GA') * (WA + SA * x))
    z = zeros(length(vars)) + -inv(H) * GA' * lambdaA
    @constraint crmodel G * z .- S * x .- W .<= 0
    @constraint crmodel lambdaA .>= 0
    crmodel
end

function simplify(poly, lib=ConvexHullLib(:float))
    polyhedron(vrep(poly), lib)
end

end