immutable Surface{D}
    position::SimpleHRepresentation{D, Float64}
    # force::SimpleHRepresentation{D, Float64}
end

# describes a single limb
immutable LimbConfig{D}
    name::String
    velocity_limit::Float64
    bounds::SimpleHRepresentation{D, Float64}
    surface::Surface{D} # the contact surface corresponding to this limb
end

immutable BoxRobot{D}
    mass::Float64
    dim::Int16
    limbs::Dict{Symbol, LimbConfig{D}}
end

immutable Environment{D}
    surfaces::Vector{Surface{D}}
end

# Now we write down things needed for simulation, namely robot state
type LimbState{T}
  pos::Vector{Float64}
  vel::Vector{Float64}
  in_contact::Bool
end

type CentroidalDynamicsState{T}
  pos::Vector{Float64}
  vel::Vector{Float64}
  # could add angular momentum later if we want
end

type BoxRobotState{T}
  centroidal_dynamics_state::CentroidalDynamicsState{T}
  limb_state::Dict{Symbol, LimbState{T}}
end

## INPUTS:
type LimbInput{T}
  acceleration::Vector{Float64}
  force::Vector{Float64}
  has_force::Bool # should have either acceleration or force, not both
end

type BoxRobotInput{T}
  limb_input::Dict{Symbol, LimbInput{T}}
end
