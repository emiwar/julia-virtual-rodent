abstract type AbstractEnv end

const RUNNING = UInt8(0)
const TRUNCATED = UInt8(1)
const TERMINATED = UInt8(2)

function state(env::AbstractEnv)
    error("state function not implemented for $(typeof(env))")
end

function info(env::AbstractEnv)
    error("info function not implemented for $(typeof(env))")
end

function reward(env::AbstractEnv)
    error("reward function not implemented for $(typeof(env))")
end

function act!(env::AbstractEnv, action)
    error("act! function not implemented for $(typeof(env))")
end

function reset!(env::AbstractEnv)
    error("reset! function not implemented for $(typeof(env))")
end

function duplicate(env::AbstractEnv)
    error("duplicate function not implemented for $(typeof(env))")
end

# Called at the beginning of each epoch. Intended to be used ny wrappers.
function prepare_epoch!(env::AbstractEnv)
    nothing
end

