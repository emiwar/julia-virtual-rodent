include("../src/utils/mujoco_quat.jl")
modelPath = "src/environments/assets/rodent_with_floor_scale080_edits.xml"
model = MuJoCo.load_model(modelPath)
d = MuJoCo.init_data(model)
MuJoCo.step!(model, d)
res = zeros(3)
qa = MuJoCo.body(d, "jaw").xquat
qb =  MuJoCo.body(d, "torso").xquat
MuJoCo.mju_subQuat(res, qa, qb)
res2 = subQuat(qa, qb)
qa
qb
res
res2

function test(qa, qb)
    res = subQuat(qa, qb)
    nothing
end


function future_quats!(fq, env::RodentImitationEnv, params)
    fqr = reshape(fq, 3, params.imitation.horizon)
    for (i, t) in enumerate(imitation_horizon(env, params))
        MuJoCo.mju_subQuat(view(fqr, :, i),
                           view(env.target.qpos, 4:7, t, env.target_clip),
                           view(env.data.qpos, 4:7))
    end
end

function future_quats_gen(env, params)
    hcat((subQuat(view(env.target.qpos, 4:7, t, env.target_clip),
                  view(env.data.qpos, 4:7)) for t=1:5)...)
end

function test(env, params)
    f = future_quats_gen(env, params)
    hcat(SVector{5}(f...)...)
end

function test2(n)
    g = Base.Generator(x->x^2, 1:n)
    SVector{n}(g...)
end

function test3()
    a = SVector(0.0, 1.0, 2.0)'
    return [a; a; a]
end
[test2(3) test2(3) test2(3)]

foo(i) = SVector(i + 0.1, i + 0.2, i + 0.3)

quad_err_at(env, t) = subQuat(view(env.target.qpos, 4:7, t, env.target_clip), view(env.data.qpos, 4:7))

@generated function future_q(env, L::Val{N}) where N
    exp = :(hcat)
    join(:(quad_err_at(env, $i)) for i=1:N, ",")
    return exp
end

@generated function SRange(L::Val{N}) where N
    return :(SVector($((:(f($i)) for i=1:N)...)))#(SVector($(1:N...)))
end

@generated function SRange(L::Val{N}, f::F) where {N, F}
    # Generate a compile-time SVector of the range 1:N, function f will be applied at runtime
    return :(SVector(($((f($i)) for i in 1:N)...)))
end