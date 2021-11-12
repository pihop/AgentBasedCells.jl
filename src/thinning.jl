mutable struct ThinningSampler <: NonHomogeneousSampling
    λ::Function
    λmax::Float64
    proposet::Union{Float64, Nothing} # previous proposal 
    tspan::Tuple{Float64, Float64}

    function ThinningSampler()
        new()
    end
end

function proposeNext!(sampler::ThinningSampler)
    U = rand(Uniform())
    sampler.proposet = sampler.tspan[1] - 1.0/sampler.λmax * log(U) 
end

function sampleFirstArrival!(sampler::ThinningSampler)
    while true
        proposeNext!(sampler)
        U = rand(Uniform())
        if sampler.proposet > sampler.tspan[end]
            sampler.proposet = nothing
            return nothing
        elseif U ≤ sampler.λ(sampler.proposet) / sampler.λmax
            return sampler.proposet
        end
    end
end

