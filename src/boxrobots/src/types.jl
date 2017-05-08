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
    dim::Int64
    gravity::Vector{Float64}
    limbs::Dict{Symbol, LimbConfig{D}}
end

immutable Environment{D}
    surfaces::Vector{Surface{D}}
end

# Now we write down things needed for simulation, namely robot state
type LimbState{T}
  pos::Vector{T}
  vel::Vector{T}
  in_contact::Bool
end

type CentroidalDynamicsState{T}
  pos::Vector{T}
  vel::Vector{T}
  # could add angular momentum later if we want
end

type BoxRobotState{T}
  centroidal_dynamics_state::CentroidalDynamicsState{T}
  limb_states::Dict{Symbol, LimbState{T}}
end

typealias ContactState Dict{Symbol, Bool}

abstract LimbInputType
immutable ConstantVelocityLimbInput <: LimbInputType end
immutable ConstantAccelerationLimbInput <: LimbInputType end


## INPUTS:
type LimbInput{T, InputType <: LimbInputType}
  input::Vector{T} # limbs are assumed to be constant velocity/acceleration
  force::Vector{T}
  has_force::Bool # should have either input or force, not both
end

function LimbInput{T, InputType}(input::AbstractVector{T}, force::AbstractVector{T}, has_force::Bool,
  ::Type{InputType})
  return LimbInput{T, InputType}(input, force, has_force)
end

type BoxRobotInput{T, InputType}
  limb_inputs::Dict{Symbol, LimbInput{T, InputType}}
end


# Trajectory: Just a container for data
type Trajectory{T}
  time::AbstractVector # sorted vector of time
  data::AbstractVector{T}
end

type ContactPlan
  plan::Trajectory{ContactState}
  next_contact_state_idx::Int64
end
