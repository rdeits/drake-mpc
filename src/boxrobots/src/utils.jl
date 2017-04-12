using Polyhedra: polyhedron, Polyhedron

function convert_vector_to_3d(v::Vector{Float64})
  """
  Converts a vector from 2d to 3d if needed, assumes the 2d vector corresponds
  to (x,z) converts it to (x,y,z) by setting y = 0
  """
  if length(v) == 3
    return v
  else
    v3d = zeros(Float64, 3)
    v3d[1] = v[1]
    v3d[3] = v[2]
    return v3d
  end
end

function polyhedron_from_bounds(args...)
  """
  makes polyhedron from bounds.
  Usage: polyehdron_from_bounds([-1,1], [-2,2])
  """
    dim = length(args)
    A = zeros(Float64, 2*dim, dim)
    b = zeros(Float64, 2*dim)

    for i=1:dim
        bounds = args[dim]
        idx_lb = 2*(i-1)+1
        idx_ub = 2*(i-1)+2

        A[idx_lb, i] = -1
        b[idx_lb] = -bounds[1]

        A[idx_ub, i] = 1
        b[idx_ub] = bounds[2]
    end

    poly = polyhedron(SimpleHRepresentation(A,b))
    return poly
end

# converts polynomial in H representation to 3d by introducing -1,1 bounds on y vars
function convert_polyhedron_to_3d(poly::Polyhedron)
  """
  Converts polyhedron in 2d to polyhedron in 3d by introducing bounds
  [-1,1] on the y component
  """
    hRep = SimpleHRepresentation(poly)
    dim = size(hRep.A)[2]


    #if dimension already 3 we are done, just return poly
    if !(dim in [2,3])
        error("dimension must be 2 or 3")
    end

    if dim == 3
        return poly
    end

    # assume it is specified in (x,z) space, lift it to x,y,z space
    A = hRep.A
    b = hRep.b
    A_size = size(A)
    ybounds = [-1,1]

    A_3d = zeros(Float64, A_size[1]+2, 3)
    A_3d[1:A_size[1],1] = A[:,1] # x coordinate
    A_3d[1:A_size[1],3] = A[:,2] # z coordinate
    A_3d[end-1, 2] = -1
    A_3d[end, 2] = 1

    b_3d = zeros(Float64, length(b) + 2)
    b_3d[1:length(b)] = b
    b_3d[end-1] = -ybounds[1]
    b_3d[end] = ybounds[2]

    poly_3d = polyhedron(SimpleHRepresentation(A_3d, b_3d))
    return poly_3d
end
