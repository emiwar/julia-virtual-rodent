import HDF5
import MuJoCo
import StaticArrays: SVector
import PythonCall
import CSV
using ProgressMeter

source_file = "/home/emil/Development/activation-analysis/local_rollouts/coltrane/2021_07_28_1/precomputed_inputs.h5"
clip_selection_file = "/home/emil/Development/example-dannce-videos/manual_selection.csv"
output_file = "/home/emil/Development/activation-analysis/local_rollouts/custom_eval1/precomputed_inputs.h5"
framerate = 50
clip_duration = 20 * framerate

full_session_ref = Dict{String, Matrix{Float64}}()
all_keys = ["position", "quaternion", "joints", "velocity", "angular_velocity", "joints_velocity",
            "center_of_mass", "appendages", "body_positions", "body_quaternions"]
HDF5.h5open(source_file, "r") do fid
    for k in all_keys
        full_session_ref[k] = fid["clip_0/walkers/walker_0/$k"] |> Array
    end
end

subclip_selection = CSV.File(clip_selection_file)
starttimes = Dates.seconds.(f.start .- Dates.Time(0))
startframes = Int.(starttimes .* framerate)
clip_notes = f.manual_notes

HDF5.h5open(output_file, "w") do fid
    for i in axes(startframes, 1)
        range = ((startframes[i]):(startframes[i] + clip_duration - 1))
        for k in all_keys
            fid["clip_$(i-1)/walkers/walker_0/$k"] = full_session_ref[k][range, :]
        end
        HDF5.attrs(fid["clip_$(i-1)"])["action"] = clip_notes[i]
    end
end
