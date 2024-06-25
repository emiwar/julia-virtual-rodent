import HDF5

function create_logger(filename, reserve_epochs, n_quantiles)
    HDF5.h5open(fid->fid["n_epochs"]=[0], filename, "cw")
    function logger(epoch, key, val)
        key = "training/$key"
        HDF5.h5open(filename, "cw") do fid
            if val isa AbstractArray
                if !haskey(fid, key)
                    HDF5.create_dataset(fid, "$key/quantiles", eltype(val), (n_quantiles, reserve_epochs))
                                       # HDF5.dataspace((0, 0), (reserve_epochs, n_quantiles));
                                       # chunk=(n_quantiles, 1))
                end
                fid["$key/quantiles"][:, epoch] = calc_quantiles(val, n_quantiles)
                #HDF5.write_attribute(fid["$key/quantiles"], "epoch", epoch)
                fid["n_epochs"][1] = epoch
            else
                if !haskey(fid, key)
                    HDF5.create_dataset(fid, key, typeof(val), (reserve_epochs,))
                end
                fid[key][epoch] = val
                #HDF5.write_attribute(fid[key], "epoch", epoch)
                fid["n_epochs"][1] = epoch
            end
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