module ConvexMomentumDynamics


using JuMP
using Gurobi
using AxisArrays
using DrakeVisualizer
using CoordinateTransformations
using Interact, Reactive
import GeometryTypes: HyperRectangle, Vec, HomogenousMesh, Point
import ColorTypes: RGBA

# type to store the info about a ContactPoint
type ContactPoint
    name::String
    rot_matrix::Array{Float64,2} # maybe should be some sort of RotationMatrix type
end

# returns the square of the L2 norm of a vector
function l2_norm(x)
    return sum(x.^2)
end

function difference_convex_functions_decomposition(l, force)
    # l = p - r, ray from com to contact location

    # the commas are important here to have size (2,) rather
    # than (1,2)
    a = [-l[3],l[2]]
    b = [l[3],-l[1]]
    c = [-l[2],-l[1]]
    d = force[2:3]
    e = force[[1,3]]
    f = force[1:2]

    # this is the decomposition of the cross product into difference
    # of convex functions
    # these are 3-vectors
    p = 1/4 * [l2_norm(a+d),l2_norm(b+e),l2_norm(c+f)]
    q = 1/4 * [l2_norm(a-d),l2_norm(b-e),l2_norm(c-f)]

    return p,q
end

# a more convenient method of using JuMP Variables than JuMP Arrays
# this is not really ready yet
macro axis_variables(m, var, axes...)
    for arg in axes
        @assert arg.head == :kw "Axis arguments must be of the form `name=domain`"
    end
    names = [arg.args[1] for arg in axes]
    domains = [arg.args[2] for arg in axes]
    ranges = [:(1:length($domain)) for domain in domains]
    axis_args = [Expr(:call, Expr(:curly, :Axis, Expr(:quote, names[i])), domains[i]) for i in eachindex(axes)]
    quote
        vars = @variable $m $var[$(ranges...)]
        $(esc(var)) = $(Expr(:call, :AxisArray, :vars, axis_args...))
    end
end


type CentroidalDynamicsVisualizer
    com # DrakeVisualizer.Visualizer
    contacts # Array{DrakeVisualize.Visualizer, 1}
end

# encodes centroidal dynamics state at single instant
type CentroidalDynamicsState
    com_position # 3 vector
    contact_locations # 3 x num_contacts
    contact_forces # 3 x num_contacts
end

function make_point(position)
    return Point(position[1], position[2], position[3])
end

# create some drawing functionality
function draw_centroidal_dynamics_state(state::CentroidalDynamicsState)
   """
    # Arguments
    * 'contact_locations': a 3 x num_contacts array specifying contact locations in world frame
    * 'contact_forces': a 3 x num_contacts array specifying contact forces in world frame

    """
    println("drawing centroidal dynamics state")
    draw_com(state.com_position)

end

function draw_contact_point_and_force(contact_location, contact_force)
    contact_ray_end = contact_location + contact_force
    sphere = HyperSphere(make_point(contact_ray_end), 0.1)
    contact_ray_end_vis = Visualizer(GeometryData(sphere, RGBA(1, 0, 0, 0.5)))
end

function draw_com(com_position)
    sphere = HyperSphere(make_point(com_position), 0.1)
    Visualizer(GeometryData(sphere, RGBA(0, 0, 1, 0.5)))
end


end
