struct BatchStepper{S, I, E}
    states::S
    rewards::Vector{Float32}
    status::Vector{UInt8}
    infos::I
    actions::Matrix{Float32}
    environments::Vector{E}
end

raw_states(mc::BatchStepper)  = data(mc.states)
raw_infos(mc::BatchStepper)   = data(mc.infos)
status(mc::BatchStepper)  = mc.status
rewards(mc::BatchStepper) = mc.rewards
actions(mc::BatchStepper) = mc.actions
n_envs(batchStepper::BatchStepper) = length(batchStepper.environments)
env_type(batchStepper::BatchStepper) = eltype(batchStepper.environments)

function BatchStepper(template_env, n_envs)
    template_state  = state(template_env, params) |> ComponentTensor
    template_info   = info(template_env, params) |> ComponentTensor
    template_action = null_action(template_env, params)

    zeros32(inds...) = zeros(Float32, inds...)
    states  = BatchComponentTensor(template_state, n_envs, array_fcn=zeros32)
    rewards = zeros(Float32, n_envs)
    status  = zeros(UInt8, n_envs)
    infos   = BatchComponentTensor(template_info, n_envs, array_fcn=zeros32)
    actions = zeros(Float32, length(template_action), n_envs)

    environments = [clone(template_env, params) for _=1:n_envs]
    BatchStepper(states, rewards, status, infos, actions, environments)
end

function prepareEpoch!(batchStepper::BatchStepper, params)
    @Threads.threads for i=1:n_envs(batchStepper)
        if params.rollout.reset_on_epoch_start
            reset!(batchStepper.environments[i], params)
        end
        batchStepper.states[:, i] = state(batchStepper.environments[i], params)
    end
end

function step!(batchStepper::BatchStepper, params, lapTimer=nothing)
    @Threads.threads for i=1:n_envs(batchStepper)
        env = batchStepper.environments[i]
        action = view(batchStepper.actions, :, i)
        act!(env, action, params)
        batchStepper.states[:, i] = state(env, params)
        batchStepper.rewards[i]  = reward(env, params)
        batchStepper.status[i]   = status(env, params)
        batchStepper.infos[:, i] = info(env, params)
        if batchStepper.status[i] != RUNNING
            reset!(env, params)
        end
    end
end

