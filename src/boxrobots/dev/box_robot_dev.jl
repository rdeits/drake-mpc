path_to_my_module = string(Base.source_dir(), "/../src/")
println(typeof(path_to_my_module))
println(LOAD_PATH)
println(path_to_my_module)
push!(LOAD_PATH, path_to_my_module)
import BoxRobots
using DrakeVisualizer


# attempt to visualize a state
DrakeVisualizer.any_open_windows() || DrakeVisualizer.new_window();
