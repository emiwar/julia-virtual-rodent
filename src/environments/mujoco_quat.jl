#Reimplementation of some functions from MuJoCo (mju_*)
#using static arrays to avoid allocations

function negQuat(quat)
    return SVector(quat[1], -quat[2], -quat[3], -quat[4])
end

function mulQuat(qa, qb)
    return SVector(
        qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3] - qa[4]*qb[4],
        qa[1]*qb[2] + qa[2]*qb[1] + qa[3]*qb[4] - qa[4]*qb[3],
        qa[1]*qb[3] - qa[2]*qb[4] + qa[3]*qb[1] + qa[4]*qb[2],
        qa[1]*qb[4] + qa[2]*qb[3] - qa[3]*qb[2] + qa[4]*qb[1]
    )
end

function normalize3(v)
    norm_v = sqrt(v[1]^2 + v[2]^2 + v[3]^2) + eps()
    return v / norm_v, norm_v
end

function quat2Vel(quat, dt)
    axis = SVector(quat[2], quat[3], quat[4])
    axis, sin_a_2 = normalize3(axis)
    speed = 2 * atan(sin_a_2 / quat[1])

    if speed > π
        speed -= 2 * π
    end

    speed /= dt
    return axis * speed
end

function subQuat(qa, qb)
    # qdif = neg(qb) * qa
    qneg = negQuat(qb)
    qdif = mulQuat(qneg, qa)

    # convert to 3D velocity (assume dt = 1 for simplicity)
    return quat2Vel(qdif, 1.0)
end
