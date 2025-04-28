module Timers

export LapTimer, lap, to_stringdict

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

function reset!(lt::LapTimer)
    for k in keys(lt.cumulTimes)
        lt.cumulTimes[k] = 0.0
    end
    lt.lapStart = NaN
    lt.lapLabel = :start
end

to_stringdict(lt::LapTimer) = Dict("timings/$k"=>v for (k,v) in pairs(lt.cumulTimes))

const default_timer = LapTimer()
lap(label::Symbol) = lap(default_timer, label)
reset!() = reset!(default_timer)

end