module Networks
    using CUDA
    using Flux
    abstract type PPOActorCritic end
    include("ppo/variational_encoder_decoder.jl")
end