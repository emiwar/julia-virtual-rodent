mutable struct Dummy
    a::Float64
end

function dummy_loss(x)
    d = Dummy(0.0)
    d.a = 1*x
    return d.a ^ 2
end

Zygote.gradient(dummy_loss, 1.0)