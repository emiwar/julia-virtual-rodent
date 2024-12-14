struct JoystickTrajectory
    forward::Vector{Float64}
    turning::Vector{Float64}
    head::Vector{Float64}
end

function JoystickTrajectory()
    traj = JoystickTrajectory(Float64[], Float64[], Float64[])
    add_static!(traj, rand(100:1000),  0.5 + rand()*0.5, 0.0, 0.05 + rand()*0.02)
    add_static!(traj, rand(20:100),   0.0, -0.5 + rand()*1.0, 0.06)
    add_static!(traj, rand(100:300), -0.2 + rand()*0.5, 0.0, 0.06)
    add_static!(traj, rand(25:50),    rand()*0.1,  0.0, 0.06)
    add_static!(traj, rand(20:100),   0.0, -1.0 + rand()*2.0, 0.06)
    add_static!(traj, rand(100:200),  0.5,  0.0, 0.05 + rand()*0.02)
    add_static!(traj, rand(50:200),   0.0,  0.0, rand()*0.02)
    add_static!(traj, rand(100:1000),  0.2 + rand()*0.5, 0.0, 0.04 + rand()*0.04)
    add_static!(traj, rand(50:75),   0.0,  0.0, 0.06)
    add_static!(traj, rand(250:500),   0.0,  0.0, 0.12)
    add_static!(traj, rand(50:75),   0.0,  0.0, 0.06)
    add_static!(traj, rand(100:500),   0.2+0.3*rand(),  0.0, 0.06)
    add_static!(traj, rand(200:400),  0.2+rand(),  0.0, 0.06)
    add_static!(traj, rand(100:400),  0.1 + rand()*0.4,  -0.5 + rand()*1.0, 0.06)
    add_static!(traj, rand(100:300),  -0.1 + rand()*0.6, 0.0, 0.06)
    add_static!(traj, rand(250:500),   0.0,  0.0, 0.08 + rand()*0.06)
    add_static!(traj, rand(20:100),   0.0, -1.0 + rand()*2.0, 0.06)
    add_static!(traj, rand(150:400),   0.0,  0.0, 0.08 + rand()*0.06)
    add_static!(traj, rand(100:1000),  0.5 + rand()*0.5, 0.0, 0.05 + rand()*0.02)
    add_static!(traj, rand(150:400),   0.05 + rand()*0.15,  0.0, 0.08 + rand()*0.06)
    add_static!(traj, rand(50:200),   0.0,  0.0, 0.05 + rand()*0.02)
    add_static!(traj, rand(250:500),   0.0,  -0.5 + rand(), 0.08 + rand()*0.06)
    add_static!(traj, rand(50:200),   0.0,  0.0, 0.05 + rand()*0.02)
    add_static!(traj, rand(250:500),   0.5 + rand()*0.5,  -0.5 + rand(), 0.05 + rand()*0.02)
    return traj
end

function add_static!(traj::JoystickTrajectory, dur, speed, turnspeed, head_height)
    for _=1:dur
        push!(traj.forward, speed)
        push!(traj.turning, turnspeed)
        push!(traj.head, head_height)
    end
end
