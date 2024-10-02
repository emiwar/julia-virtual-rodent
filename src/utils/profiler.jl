mutable struct LapTimer
    lapStart::Float64
    lapLabel::Symbol
    cumulTimes::Dict{Symbol, Float64}
end

LapTimer() = LapTimer(NaN, :start, Dict{Symbol, Float64}())

function lap(lt::LapTimer, label::Symbol)
    t = time()
    if isfinite(lt.lapStart)
        if !haskey(lt.cumulTimes, lt.lapLabel)
            lt.cumulTimes[lt.lapLabel] = 0.0
        end
        lt.cumulTimes[lt.lapLabel] += t - lt.lapStart
    end
    lt.lapStart = t
    lt.lapLabel = label
    return nothing
end

to_stringdict(lt::LapTimer) = Dict("timings/$k"=>v for (k,v) in pairs(lt.cumulTimes))
