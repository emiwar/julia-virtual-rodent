struct ImitationTarget
    qpos::Matrix{Float64}
    qvel::Matrix{Float64}
    com::Matrix{Float64}
    xquat::Matrix{Float64}
    xmat::Matrix{Float64}
end

function ImitationTarget(model)
    trajectoryPath = "src/environments/assets/com_trajectory2.h5"
    qpos, com, xquat, xmat = HDF5.h5open(trajectoryPath, "r") do fid
        map(k->fid[k][:, :], ("qpos", "com", "xquat", "xmat"))
    end
    qvel = estimate_qvel(model, qpos, dt=0.002)
    ImitationTarget(qpos, qvel, com, xquat, xmat)
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