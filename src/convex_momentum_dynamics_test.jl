include("ConvexMomentumDynamics.jl")
using ConvexMomentumDynamics

com_position = [0.,0.,1.]
draw_com(com_position)

com_position = [0.,0.,1.]
num_contacts = 2
contact_location_test = zeros(3,num_contacts)
contact_location_test[:,1] = [-0.5,0,0]
contact_location_test[:,2] = [0.5,0,0]
display(contact_location_test)

contact_forces = similar(contact_location_test)
contact_forces[:,1] = [0,0,0.5]
contact_forces[:,2] = [0,0,0.5]

state = CentroidalDynamicsState(com_position, contact_location_test, contact_forces)
display(state.com_position)
# draw_centroidal_dynamics_state(state)


draw_centroidal_dynamics_state(state)

contact_names = ["left_foot", "right_foot"]
num_contacts = length(contact_names)


contact_name_to_idx = Dict()
for (idx, name) in enumerate(contact_names)
    contact_name_to_idx[name] = idx
end

# initialize constants
num_timesteps = 5
num_knot_points = num_timesteps + 1
robot_mass = 1
dt = 0.1
gravity = [0,0,-9.8]
dimension = 3

#setup foot positions in world
contact_location = Dict()
contact_location["left_foot"] = [-0.5,0,0]
contact_location["right_foot"] = [0.5,0,0]
mu = 0.5


# com_data
com_data = Dict();
com_data["initial_position"] = [0,0,1]
com_data["final_position"] = [0,0,1]


# weights
weights = Dict()
weights["com_final_position"] = 100
weights["lin_momentum"] = 0.1
weights["forces"] = 0.01
weights["convex_bounds"] = 1e-4 # if this is too large forces will just be zero

# initialize the model
model = Model()


# add variables to model

# com position
@variable(model, com_position[1:3, 1:num_knot_points])

# linear momentum
@variable(model, lin_momentum[1:3, 1:num_knot_points])

# angular momentum
@variable(model, ang_momentum[1:3, 1:num_knot_points])

# linear momentum dot
@variable(model, lin_momentum_dot[1:3, 1:num_knot_points])

# angular momentum dot
@variable(model, ang_momentum_dot[1:3,1:num_knot_points])

# forces
contact_list = ["left_foot", "right_foot"]
@variable(model, forces[1:3, 1:num_timesteps, 1:num_contacts] >= 0)

# will eventually need something like frames attached to those bodies so we can figure
# the friction cone and stuff
# contact_frames would be an array of rotation matrices basically that
# transform body to world


# torques
# bound them at zero for now
@variable(model, 0 <= torques[1:3, 1:num_timesteps, 1:num_contacts] <= 0)

# bounding variables for l x f terms
@variable(model, l_cross_f_plus[1:dimension, 1:num_timesteps, 1:num_contacts] >= 0)
@variable(model, l_cross_f_minus[1:dimension, 1:num_timesteps, 1:num_contacts] >= 0)

# add constraints to model
for i=1:num_timesteps

    # com position integration constraint
    @constraint(model, com_position[:,i+1] .- (com_position[:,i] .+
        dt/robot_mass.*lin_momentum[:,i]) .== 0)

    # linear momentum integration constraint
    @constraint(model, lin_momentum[:,i+1] .- (lin_momentum[:,i]
            + lin_momentum_dot[:,i]*dt) .== 0)

    # angular momentum integration constraint
    @constraint(model, ang_momentum[:, i+1] .- (ang_momentum[:,i]
            .+ ang_momentum_dot[:,i] * dt) .== 0)


    total_force = zero(AffExpr) # total force
    total_torque = zero(AffExpr) # total torque

    for contact_idx =1:num_contacts
        total_force += forces[:,i,contact_idx]
        total_torque += torques[:,i,contact_idx]
    end

    # F = ma
    @constraint(model, lin_momentum_dot[:,i] - (robot_mass*gravity .+ total_force) .== 0)

    # the total l x f term
    l_cross_f = zero(AffExpr)
    for contact_idx =1:num_contacts

        # lookup string name of contact
        contact_name = contact_names[contact_idx]

        # vector from com --> contact location
        l = contact_location[contact_name] .- com_position[:,i]
        force_local = forces[:,i,contact_idx]

        # decompose into difference of convex functions
        p,q = difference_convex_functions_decomposition(l, force_local)

        # add constraint for convex relaxation
        @constraint(model, p .<= l_cross_f_plus[:,i,contact_idx])
        @constraint(model, q .<= l_cross_f_minus[:,i,contact_idx])

        # add the contribution of this contact to the overally l x f term.
        l_cross_f += l_cross_f_plus[:,i,contact_idx] - l_cross_f_minus[:,i,contact_idx]
    end

    # angular momentum dot constraint
    @constraint(model, ang_momentum_dot[:,i] .- (l_cross_f .+ total_torque) .== 0)
end

# initial condition constraints
@constraint(model, com_position[:,1] .== com_data["initial_position"])
@constraint(model, lin_momentum[:,1] .== zeros(3))
@constraint(model, lin_momentum_dot[:,1] .== zeros(3))

# add objective
com_final_position_cost = weights["com_final_position"]*
l2_norm(com_position[:,end] - com_data["final_position"])

lin_momentum_cost = weights["lin_momentum"]*l2_norm(lin_momentum[:,:,:])

force_cost = zero(QuadExpr)
convex_bound_cost = zero(QuadExpr)
for contact_idx=1:num_contacts
    force_cost += weights["forces"]*l2_norm(forces[:,:,contact_idx])
    convex_bound_cost += weights["convex_bounds"]*l2_norm(l_cross_f_plus[:,:,contact_idx])
    convex_bound_cost += weights["convex_bounds"]*l2_norm(l_cross_f_minus[:,:,contact_idx])
end

total_objective = com_final_position_cost + lin_momentum_cost
+ force_cost

total_objective += convex_bound_cost

@objective(model, Min, total_objective)

# solve the model
status = solve(model)

# parse solution data
# trailing underscore indicates value of those elements
com_position_ = getvalue(com_position)
lin_momentum_ = getvalue(lin_momentum)
ang_momentum_ = getvalue(ang_momentum)
lin_momentum_dot_ = getvalue(lin_momentum_dot)
ang_momentum_dot_ = getvalue(ang_momentum_dot)
l_cross_f_plus_ = getvalue(l_cross_f_plus)
l_cross_f_minus_ = getvalue(l_cross_f_minus)
forces_ = getvalue(forces)

p_ = zeros(l_cross_f_plus_)
q_ = zeros(l_cross_f_plus_)

# populate q_, v_ check tightness with l_cross_f_plus/minus
for i=1:num_timesteps
    for contact_idx = 1:num_contacts
        contact_name = contact_names[contact_idx]
        l = contact_location[contact_name] - com_position_[:,i]
        force_local = forces_[:,i,contact_idx]
        p_[:,i,contact_idx], q_[:,i,contact_idx] =
        difference_convex_functions_decomposition(l,force_local)
    end
end

p_slack = l_cross_f_plus_ - p_
display(p_slack)
