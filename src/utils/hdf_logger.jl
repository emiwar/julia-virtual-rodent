import HDF5
import Sockets

function create_logger(filename, reserve_epochs, n_quantiles; buffer_size=16)
    HDF5.h5open(fid->fid["n_epochs"]=[0], filename, "cw")
    buffer = Dict{String, Array}()
    last_write = Ref(0)
    function logger(epoch, key, val)
        key = "training/$key"

        if epoch-last_write[] > buffer_size
            HDF5.h5open(filename, "cw") do fid
                for (k, v) in pairs(buffer)
                    if ndims(v) == 2
                        if !haskey(fid, k)
                            HDF5.create_dataset(fid, k, eltype(v), (n_quantiles, reserve_epochs))
                        end
                        fid[k][:, (last_write[]+1):(epoch-1)] = v
                    else
                        if !haskey(fid, k)
                            HDF5.create_dataset(fid, k, eltype(v), (reserve_epochs,))
                        end
                        fid[k][(last_write[]+1):(epoch-1)] = v
                    end
                end
                fid["n_epochs"][1] = epoch - 1
            end
            last_write[] = epoch-1
            for v in values(buffer)
                v .= 0.0
            end
        end

        if val isa AbstractArray
            key = "$key/quantiles"
            if !haskey(buffer, key)
                buffer[key] = zeros(n_quantiles, buffer_size)
            end
            buffer[key][:, epoch-last_write[]] = calc_quantiles(val, n_quantiles)
        else
            if !haskey(buffer, key)
                buffer[key] = zeros(buffer_size)
            end
            buffer[key][epoch-last_write[]] = val
        end
        
    end
end

function calc_quantiles(vals, n_quantiles)
    if length(vals) <= n_quantiles
        pad_l = Int(ceil((n_quantiles - length(vals)) / 2))
        pad_r = Int(floor((n_quantiles - length(vals)) / 2))
        return Array([fill(NaN, pad_l); sort(Array(reshape(vals, :))); fill(NaN, pad_r)])
    elseif length(vals) % n_quantiles == 0
        stepsize = length(vals) ÷ n_quantiles
        return Array(Float64.(view(sort(view(vals, :)), (stepsize÷2):stepsize:length(vals))))
    else
        #Inefficient since need to copy
        return Array(Statistics.quantile(Array(view(vals, :)), ((1:n_quantiles) .- 0.5) ./ n_quantiles))
    end
end

function write_params(filename::String, params)
    HDF5.h5open(filename, "cw") do fid
        for (k,v) in pairs(params)
            fid["params/$k"] = v
        end
    end
end

function write_sysinfo(filename::String)
    HDF5.h5open(filename, "cw") do fid
        fid["system/hostname"] = Sockets.gethostname()
        fid["system/git"] = readchomp(`git rev-parse --short HEAD`)
    end
end