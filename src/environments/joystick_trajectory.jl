struct JoystickTrajectory
    forward::Vector{Float64}
    turning::Vector{Float64}
end

function JoystickTrajectory()
    traj = JoystickTrajectory(Float64[], Float64[])
    add_static!(traj, rand(100:300),  0.1 + rand()*3.0, 0.0)
    add_static!(traj, rand(20:100),   0.0, -5.0 + rand()*10.0)
    add_static!(traj, rand(100:300), -0.5 + rand()*0.4, 0.0)
    add_static!(traj, rand(25:50),    1.0,  0.0)
    add_static!(traj, rand(20:100),   0.0, -5.0 + rand()*10.0)
    add_static!(traj, rand(100:200),  3.0,  0.0)
    add_static!(traj, rand(100:200),  0.0,  0.0)
    add_static!(traj, rand(100:400),  0.5 + rand()*3.0,  -3.0 + rand()*6.0)
    add_static!(traj, rand(100:300),  -0.5 + rand()*2.0, 0.0)
    add_static!(traj, rand(20:100),   0.0, -5.0 + rand()*10.0)
    return traj
end

function add_static!(traj::JoystickTrajectory, dur, speed, turnspeed)
    for _=1:dur
        push!(traj.forward, speed)
        push!(traj.turning, turnspeed)
    end
end
