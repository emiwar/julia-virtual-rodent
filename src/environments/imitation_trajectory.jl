import HDF5

struct ImitationTarget
    qpos::Array{Float64, 3}
    qvel::Array{Float64, 3}
    com::Array{Float64, 3}
end

function ImitationTarget()
    qpos, qvel, com = HDF5.h5open("src/environments/assets/diego_curated_snippets.h5", "r") do fid
        n_clips = length(keys(fid))
        qpos = zeros(74, 250, n_clips)
        qvel = zeros(73, 250, n_clips)
        com = zeros(3, 250, n_clips)
        for clip_id = 0:(n_clips-1)
            qpos[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/position"])'
            qpos[4:7, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/quaternion"])'
            qpos[8:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints"])'
            qvel[1:3, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/velocity"])'
            qvel[4:6, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/angular_velocity"])'
            qvel[7:end, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/joints_velocity"])'
            com[:, :, clip_id+1] .= read(fid["clip_$clip_id/walkers/walker_0/center_of_mass"])'
        end
        return qpos, qvel, com
    end
    ImitationTarget(qpos, qvel, com)
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



