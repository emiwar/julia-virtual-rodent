import HDF5
import MuJoCo
using ProgressMeter
include("../src/environments/imitation_trajectory.jl")
HDF5.h5open(display, "src/environments/assets/diego_curated_snippets.h5", "r")
HDF5.h5open(f->size(f["clip_0/walkers/walker_0/body_quaternions"]), "src/environments/assets/diego_curated_snippets.h5", "r")

function infer_com_and_appendages(model, qpos)
    T = size(qpos, 2)
    center_of_mass = zeros(3, T)
    appendages_pos = zeros(3, length(appendages_order()), T)
    body_pos = zeros(3, length(bodies_order()), T)
    body_quat = zeros(4, length(bodies_order()), T)
    d = MuJoCo.init_data(model)
    @showprogress for t = 1:T
        d.qpos .= @view qpos[:, t]
        MuJoCo.forward!(model, d)
        torso = MuJoCo.body(d, "walker/torso")
        torso_xmat = reshape(torso.xmat, 3, 3)
        center_of_mass[:, t] .= torso.com
        for (i, app_name) in enumerate(appendages_order())
            app_body = MuJoCo.body(d, "walker/$app_name")
            appendages_pos[:, i, t] = torso_xmat*(app_body.xpos .- torso.xpos)
        end
        for (i, body_name) in enumerate(bodies_order())
            body = MuJoCo.body(d, "walker/$body_name")
            body_pos[:, i, t] = body.xpos
            body_quat[:, i, t] = body.xquat
        end
    end
    return center_of_mass, appendages_pos, body_pos, body_quat
end

function estimate_qvel(model, qpos; dt)
    T = size(qpos, 2)
    output = zeros(model.nv)
    qvel = zeros(model.nv, T)
    for t=1:(T-1)
        MuJoCo.mj_differentiatePos(model, output, dt, qpos[:, t], qpos[:, t+1])
        qvel[:, t] .= clamp.(output, -20, 20) #Seems Diego is clamping like this?
    end
    return qvel
end


#sess_name = "2020_12_22_1"
#qpos = HDF5.h5open(f->read(f["pose/qpos"]), "reference_data/$(sess_name).h5", "r")
qpos = array(load_imitation_target().qpos)[:, :, 1]


model = dm_control_rodent(torque_actuators = true,
                          foot_mods = true,
                          scale = 1.0,
                          hip_mods = false,
                          physics_timestep = 0.002,
                          control_timestep = 0.01)
qvel = estimate_qvel(model, qpos; dt=1.0/50.0);
center_of_mass, appendages, body_pos, body_quat = infer_com_and_appendages(model, qpos);
HDF5.h5open("reference_data/clip0_recomputed.h5", "w") do fid
    fid["clip_0/walkers/walker_0/position"] = qpos[1:3, :]' |> collect
    fid["clip_0/walkers/walker_0/quaternion"] = qpos[4:7, :]' |> collect
    fid["clip_0/walkers/walker_0/joints"] = qpos[8:end, :]' |> collect

    fid["clip_0/walkers/walker_0/velocity"] = qvel[1:3, :]' |> collect
    fid["clip_0/walkers/walker_0/angular_velocity"] = qvel[4:6, :]' |> collect
    fid["clip_0/walkers/walker_0/joints_velocity"] = qvel[7:end, :]' |> collect

    fid["clip_0/walkers/walker_0/center_of_mass"] = center_of_mass' |> collect
    fid["clip_0/walkers/walker_0/appendages"] = reshape(appendages, :, size(appendages, 3))' |> collect
    fid["clip_0/walkers/walker_0/body_positions"] = reshape(body_pos, :, size(body_pos, 3))' |> collect
    fid["clip_0/walkers/walker_0/body_quaternions"] = reshape(body_quat, :, size(body_quat, 3))' |> collect
end

target = load_imitation_target("reference_data/$(sess_name)_precomputed.h5")


new = load_imitation_target("reference_data/clip0_recomputed.h5");
old = @view load_imitation_target()[:, :, 1:1];
