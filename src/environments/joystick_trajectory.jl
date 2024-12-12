struct JoystickTrajectory
    forward::Vector{Float64}
    turning::Vector{Float64}
end

function JoystickTrajectory()
    traj = JoystickTrajectory(Float64[], Float64[])
    if rand() > 0.5
        add_static!(traj, rand(500:1000),  0.5 + rand()*0.5, 0.0)
    end
    add_static!(traj, rand(20:100),   0.0, -5.0 + rand()*10.0)
    add_static!(traj, rand(100:300), -0.2 + rand()*0.3, 0.0)
    add_static!(traj, rand(25:50),    0.3,  0.0)
    add_static!(traj, rand(20:100),   0.0, -4.0 + rand()*8.0)
    add_static!(traj, rand(100:200),  0.5,  0.0)
    add_static!(traj, rand(100:200),  0.0,  0.0)
    add_static!(traj, rand(100:400),  0.1 + rand()*0.4,  -3.0 + rand()*6.0)
    add_static!(traj, rand(100:300),  -0.1 + rand()*0.6, 0.0)
    add_static!(traj, rand(20:100),   0.0, -5.0 + rand()*10.0)
    return traj
end

function add_static!(traj::JoystickTrajectory, dur, speed, turnspeed)
    for _=1:dur
        push!(traj.forward, speed)
        push!(traj.turning, turnspeed)
    end
end
