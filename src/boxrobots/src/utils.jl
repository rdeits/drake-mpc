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
