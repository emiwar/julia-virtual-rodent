abstract type Walker end


null_action(walker::Walker) = zeros(walker.model.nu)

function create_sensorindex(model::MuJoCo.Model)
    map(1:model.nsensor) do i
        start = model.sensor_adr[i] + 1
        len  = model.sensor_dim[i]
        stop = start + len - 1
        name = Symbol(MuJoCo.mj_id2name(model, MuJoCo.mjOBJ_SENSOR, i-1) |> unsafe_string)
        if len==1
            return name=>start
        else 
            name=>ComponentArrays.ViewAxis(start:stop, ComponentArrays.Shaped1DAxis((Int64(len),)))
        end
    end |> NamedTuple |> ComponentArrays.Axis
end

function create_actuatorindex(model::MuJoCo.Model)
    map(1:model.na) do i
        start = model.actuator_actadr[i] + 1
        len  = model.actuator_actnum[i]
        stop = start + len - 1
        name = Symbol(MuJoCo.mj_id2name(model, MuJoCo.mjOBJ_ACTUATOR, i-1) |> unsafe_string)
        if len==1
            return name=>start
        else 
            name=>ComponentArrays.ViewAxis(start:stop, ComponentArrays.Shaped1DAxis((Int64(len),)))
        end
    end |> NamedTuple |> ComponentArrays.Axis
end

function subtree_com(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{3}(view(walker.data.subtree_com, ind, :))
end

function body_xmat(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SMatrix{3, 3}(reshape(view(walker.data.xmat, ind, :), 3, 3))
end

function body_xpos(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{3}(view(walker.data.xpos, ind, :))
end

function body_xquat(walker::Walker, body::String)
    ind = MuJoCo.NamedAccess.index_by_name(walker.data, MuJoCo.mjOBJ_BODY, body)+1
    SVector{4}(view(walker.data.xquat, ind, :))
end

sensor(walker::Walker, sensorname::String) = sensor(walker, Symbol(sensorname))
sensor(walker::Walker, sensorname::Symbol) = sensor(walker, Val(sensorname))
sensor(walker::Walker, sensorname::Val) = view(walker.sensors, sensorname)

function qpos_root(walker::Walker)
    return @view walker.data.qpos[1:7]
end

function joint_indices(walker::Walker)
    return Int32(8):(walker.model.nq::Int32)
end

function joint_vel_indices(walker::Walker)
    return Int32(7):(walker.model.nv::Int32)
end

function step!(walker::Walker)
    MuJoCo.step!(walker.model, walker.data)
end

function reset!(walker::Walker)
    MuJoCo.reset!(walker.model, walker.data)
    MuJoCo.forward!(walker.model, walker.data)
end

function set_ctrl!(walker::Walker, action)
    walker.data.ctrl .= clamp.(action, -1.0, 1.0)
end

function info(walker::Walker)
    (
     qpos_root=qpos_root(walker),
     torso_x=torso_x(walker),
     torso_y=torso_y(walker),
     torso_z=torso_z(walker),
     actuator_force_sum_sqr = norm(walker.data.actuator_force)^2,
     energy_use = energy_use(walker)
    )
end

function energy_use(walker::Walker)
    mapreduce((v,f) -> abs(v)*abs(f), +,
              (@view walker.data.qvel[7:end]),
              (@view walker.data.qfrc_actuator[7:end]))
end

dt(walker::Walker) = walker.model.opt.timestep

function joint_angle(walker::Walker, name::String)
    joint_id = MuJoCo.mj_name2id(walker.model, MuJoCo.mjOBJ_JOINT, name)
    if joint_id < 0
        throw(KeyError(name))
    end
    ind = walker.model.jnt_qposadr[joint_id + 1] + 1 #MuJoCo (C) is 0-indexed, Julia is 1-indexed
    return walker.data.qpos[ind] 
end

