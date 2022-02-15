mutable struct StochasticDilutionModel
    steady_state::Vector{Float64}
    experiment::AbstractExperimentSetup

    function StochasticDilutionModel(experiment)
        new([], experiment)
    end
end

struct BirthDeathException <: Exception end
Base.showerror(io::IO, e::BirthDeathException) = print("Not a birth death process.")

function is_birth_death(rn::ReactionSystem)
    # Do we care if either birth or death process doesn't exist?
    return length(Catalyst.species(rn)) == 1
end

function birth_death_ss!(model::StochasticDilutionModel)
    # Before we do anything check that the rn is a birth death process.
    rn = model.experiment.effective_dilution
    truncation = prod(model.experiment.truncation)
    model_parameters = model.experiment.model_parameters
    state = model.experiment.init

    if !is_birth_death(rn)
        throw(BirthDeathException())
    end

    reacts = reactions(rn)
    rates = [r.rate for r in reacts]
    stoich = Catalyst.netstoichmat(rn)

    ratef = gen_division_rate_function.(rates, rn)
    eval_rates = [[f(state, model_parameters, 0.0) for f in ratef] for state in 1:truncation] 
    # currently evaluated for t = 0.0. Ok if rates are not time dependent.
    sumλ = sum.(map(x -> x[vec(stoich .== 1)], eval_rates))
    sumμ = sum.(map(x -> x[vec(stoich .== -1)], eval_rates)) .* (collect(1:truncation) .- 1)

    cprods = cumprod([λ/sumμ[i+1] for (i, λ) in enumerate(sumλ[1:end-1])])
    
    π₀ = 1 / (1 + sum(cprods))
    πₖ = π₀ .* cprods 
    model.steady_state = vcat(π₀, πₖ)
end

function mean_steady_state(model::StochasticDilutionModel)
    return sum(model.steady_state .* (collect(1:length(model.steady_state)) .- 1))

end
