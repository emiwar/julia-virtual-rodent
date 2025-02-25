"""
    GaussianActionSampler(; sigma_min=0.0f0, sigma_max=1.0f0)
    Create a Gaussian action sampler with the given minimum and maximum standard deviation.
    Input to the sampler is an array with first dim 2*action_size, where the first half is the mean 
    and the second half is the unscaled standard deviation. Both action and unscaled standard 
    deviation are assumed to be in the range -1 to 1. An action can optionally be provided, in which 
    case the sampler will not draw a new action but return the loglikelihood of that action together 
    with the loglikelihood.
"""
function GaussianActionSampler(; sigma_min=0.0f0, sigma_max=1.0f0)
    #Assume unscaled sigma is in [-1, 1]
    sigma_scale  = 0.5f0 * (sigma_max - sigma_min)
    sigma_offset = 0.5f0 * (sigma_min + sigma_max)

    function sample(mu_and_sigma, action=nothing)
        mu, unscaled_sigma = split_halfway(mu_and_sigma; dim=1)
        sigma = sigma_offset .+ sigma_scale .* unscaled_sigma
        if isnothing(action)
            xsi = randn_like(mu)
            action = mu .+ sigma .* xsi
        end
        loglikelihood = -0.5f0 .* sum(((action .- mu) ./ sigma).^2; dims=1) .- sum(log.(sigma); dims=1)
        entropy_loss = 0.5sum(log.((2π*exp(1)).*(sigma.^2)); dims=1) / size(sigma, 1)
        return (;action, mu, sigma, loglikelihood, entropy_loss)
    end
end
