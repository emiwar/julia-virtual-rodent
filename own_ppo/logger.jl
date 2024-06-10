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
                fid["$key/quantiles"][:, epoch] = calc_quantiles(val, n_quantiles) |> Flux.cpu
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
    stepsize = length(vals) ÷ n_quantiles
    view(sort(view(vals, :)), (stepsize÷2):stepsize:length(vals))
end