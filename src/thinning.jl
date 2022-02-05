mutable struct ThinningSampler <: NonHomogeneousSampling
    λ::Function
    λmax::Float64
    proposet::Union{Float64, Nothing} # previous proposal 
    tspan::Tuple{Float64, Float64}

    function ThinningSampler()
        new()
    end
end

function propose_next!(sampler::ThinningSampler)
    U = rand(Uniform())
    sampler.proposet = sampler.tspan[1] - 1.0/sampler.λmax * log(U) 
end

function sample_first_arrival!(sampler::ThinningSampler)
    while true
        propose_next!(sampler)
        U = rand(Uniform())
        if sampler.proposet > sampler.tspan[end]
            sampler.proposet = nothing
            return nothing
        elseif U ≤ sampler.λ(sampler.proposet) / sampler.λmax
            return sampler.proposet
        end
    end
end

function sample_next_division(
    cell_trajectory,
    tspan,
    model::AbstractSimulationModel,
    sampler::ThinningSampler)

    # Aim is to return the first jump time of NHPP with rate λ(t, f(t)) where
    # f(t) is the CellTrajectory.
    # How do we find λ such that λ ≥ λ(t, f(t)) for all t in general.
    # f(t) in our case monotonic so just take the end point of tspan.
    
    sampler.λmax = model.model.division_rate(
        cell_trajectory(tspan[end]), 
        parameters(model.experiment), 
        tspan[end])

    sampler.λ = t -> model.model.division_rate(cell_trajectory(t), parameters(model.experiment), t)
    sampler.tspan = tspan 
    t = sample_first_arrival!(sampler)
    return t 
end


