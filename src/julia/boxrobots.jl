module BoxRobots

using Polyhedra: SimpleHRepresentation

immutable LimbConfig{D}
    velocity_limit::Float64
    bounds::SimpleHRepresentation{D, Float64}
end

immutable BoxAtlas{D}
    mass::Float64
    limbs::Dict{Symbol, LimbConfig{D}}
end

immutable Surface{D}
    pose_constraints::SimpleHRepresentation{D, Float64}
    force_constraints::SimpleHRepresentation{D, Float64}
end

immutable Environment{D}
    surfaces::Vector{Surface{D}}
end

end
