include("networks/networks.jl")
include("environments/mujoco_env.jl")
include("algorithms/mpi_ppo/mpi_ppo.jl")

include("params.jl")
template_env = MuJoCoEnvs.RodentImitationEnv(params)

function just_state(template_env, params)
    MuJoCoEnvs.state(template_env, params)
    nothing
end