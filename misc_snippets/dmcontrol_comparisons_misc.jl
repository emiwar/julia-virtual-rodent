using PyCall
import Distributions

dm_control_rodent = pyimport("dm_control.locomotion.examples.basic_rodent_2020")

dm_control_env = dm_control_rodent.rodent_run_gaps()

action_spec = dm_control_env.action_spec()
time_step = dm_control_env.reset()
action = zeros(38)#rand(Distributions.Uniform(-1, 1), 38)
time_step = dm_control_env.step(action)

import MuJoCo
modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"

model = MuJoCo.load_model(modelPath)
model.opt.timestep = dm_control_rodent._PHYSICS_TIMESTEP
data = MuJoCo.init_data(model)
MuJoCo.reset!(model, data)
MuJoCo.forward!(model, data)
_, _, _, dm_control_obs = dm_control_env.reset()
dm_control_obs["walker/appendages_pos"] ≈ get_appendage_pos(data)

_, _, _, dm_control_obs = dm_control_env.step(action)
dm_control_like_step!(model, data, action)
dm_control_obs["walker/appendages_pos"] ≈ get_appendage_pos(data)

function get_appendage_pos(data)
    appendage_names = ["lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"]
    appendages = [MuJoCo.body(data, a) for a=appendage_names]
    torso = MuJoCo.body(data, "torso")

    relative_pos = (reshape(torso.xmat, 3, 3)' * (a.xpos - torso.xpos) for a in appendages)
    relative_pos = cat(relative_pos...; dims=1)
    return relative_pos
end

function dm_control_like_step!(model, data, action)
    data.ctrl .= action
    for _=1:20
        MuJoCo.step!(model, data)
    end
end

#[reshape(torso.xmat, 3, 3) 
#      end_effectors_with_head = (self._entity.end_effectors + (self._entity.head,))
#      end_effector = physics.bind(end_effectors_with_head).xpos
#      torso = physics.bind(self._entity.root_body).xpos
#      xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
#      return np.reshape(np.dot(end_effector - torso, xmat), -1)


function step_envs(envs)
    #@Threads.threads 
    for env in envs
        action = rand(Distributions.Uniform(-1, 1), 38)
        env.step(action)
    end
end

function create_envs(nenvs)
    envs = [dm_control_rodent.rodent_two_touch() for _=1:nenvs]
    for env in envs
        env.reset()
    end
    return envs
end

envs = create_envs(20)

step_envs(envs)



