module MPI_PPO
import BSON
import CUDA
import MPI
using Flux
using ProgressMeter
#include("../../utils/component_tensor.jl")
include("../../utils/profiler.jl")
include("../../utils/wandb_logger.jl")
#using .ComponentTensors

include("main_loop.jl")
include("collector.jl")

end
