import HDF5
import MuJoCo
using ProgressMeter

struct ImitationTarget
    #TODO: Use ComponentTensors for this
    #(and maybe store on GPU for faster access in state(...)?)
    qpos::Array{Float64, 3}
    qvel::Array{Float64, 3}
    com::Array{Float64, 3}
    appendages::Array{Float64, 4}
end

function ImitationTarget(model)
    qpos, qvel = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
        n_clips = length(keys(fid))
        qpos = zeros(74, 250, n_clips)
        qvel = zeros(73, 250, n_clips)
        #com = zeros(3, 250, n_clips)
        for clip_id = 0:(n_clips-1)
            qpos[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/position"])'
            qpos[4:7, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/quaternion"])'
            qpos[8:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints"])'
            qvel[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/velocity"])'
            qvel[4:6, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/angular_velocity"])'
            qvel[7:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints_velocity"])'
            #com[:, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/center_of_mass"])'
        end
        return qpos, qvel#, com
    end
    actuation = zeros(model.na, size(qpos)[2:3]...)
    fullphysics = cat(qpos, qvel, actuation; dims=1)
    center_of_mass, appendages_pos = infer_com_and_appendages(model, fullphysics)
    ImitationTarget(qpos, qvel, center_of_mass, appendages_pos)
end

appendages() = ("walker/lower_arm_R", "walker/lower_arm_L", "walker/foot_R", "walker/foot_L", "walker/jaw")

function infer_com_and_appendages(model, fullphysics)
    _, T, n_clips = size(fullphysics)
    center_of_mass = zeros(3, T, n_clips)
    appendages_pos = zeros(3, 5, T, n_clips)
    @Threads.threads for clip_id = 1:n_clips
        d = MuJoCo.init_data(model)
        for t = 1:T
            MuJoCo.set_physics_state!(model, d, view(fullphysics, :, t, clip_id))
            MuJoCo.forward!(model, d)
            torso = MuJoCo.body(d, "walker/torso")
            torso_xmat = reshape(torso.xmat, 3, 3)
            center_of_mass[:, t, clip_id] .= torso.com
            for (i, app_name) in enumerate(appendages())
                app_body = MuJoCo.body(d, app_name)
                appendages_pos[:, i, t, clip_id] = torso_xmat*(app_body.xpos .- torso.xpos)
            end
        end
    end
    return center_of_mass, appendages_pos
end

function estimate_qvel(model, qpos; dt)
    T = size(qpos, 2)
    output = zeros(model.nv)
    qvel = zeros(model.nv, T)
    for t=2:T
        MuJoCo.mj_differentiatePos(model, output, dt, qpos[:, t-1], qpos[:, t])
        qvel[:, t] .= output
    end
    return qvel
end

Base.length(target::ImitationTarget) = size(target.qpos, 2)



