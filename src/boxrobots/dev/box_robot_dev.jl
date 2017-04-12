path_to_my_module = string(Base.source_dir(), "/../src/")
push!(LOAD_PATH, path_to_my_module)

using BoxRobots
using DrakeVisualizer
import ColorTypes: RGBA

# attempt to visualize a state
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
vis = Visualizer()

# test constructing and drawing a polyhedron
poly = polyhedron_from_bounds([-1,1], [-1,1])
poly_3d = convert_polyhedron_to_3d(poly)
green = RGBA(0.,1.0,0.0)

setgeometry!(vis[:poly], GeometryData(poly_3d, green))
