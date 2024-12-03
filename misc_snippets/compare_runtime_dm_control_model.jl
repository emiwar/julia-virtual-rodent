import MuJoCo
import PythonCall

include("../src/utils/load_dm_control_model.jl")
MuJoCo.init_visualiser()

function step1000(model, data)
    for i=1:1000
        MuJoCo.step!(model, data)
    end
end

function print_all_contacts(data)
    if isnothing(data.contact)
        println("<No contacts>")
        return
    end
    for contact in data.contact'
        A = unsafe_string(MuJoCo.mj_id2name(model, MuJoCo.mjOBJ_GEOM, contact.geom1))
        B = unsafe_string(MuJoCo.mj_id2name(model, MuJoCo.mjOBJ_GEOM, contact.geom2))
        println("$(A) -> $(B)")
    end
end

#DM Control
model = dm_control_rodent(torque_actuators = true, foot_mods = false,
                          scale = 1.0, hip_mods = false, physics_timestep = 0.002,
                          control_timestep = 0.01)
data = MuJoCo.init_data(model)
MuJoCo.reset!(model, data)
#data.qpos[3] += 0.002
@time step1000(model, data)
print_all_contacts(data)
MuJoCo.visualise!(model, data)

model = MuJoCo.load_model("src/environments/assets/rodent_with_floor_scale080_torques.xml")
data = MuJoCo.init_data(model)
MuJoCo.reset!(model, data)
data.qpos[3] += 0.002
@time step1000(model, data)
print_all_contacts(data)
MuJoCo.visualise!(model, data)