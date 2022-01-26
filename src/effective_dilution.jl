mutable struct EffectiveDilutionModel
    roots_function::Function
    search_interval::Interval{Float64}
    parameters::Vector{Float64}
    parameter_span::StepRangeLen
    bifurcation_index::Int64
    roots
    experiment::AbstractExperimentSetup

    function EffectiveDilutionModel(roots_function, search_interval, bifurcation_index, parameter_span, experiment)
        new(roots_function, search_interval, experiment.model_parameters, parameter_span, bifurcation_index, [], experiment)
    end
end

function root_finding!(model::EffectiveDilutionModel)
    params = [] 
    for p in model.parameter_span
        _p = copy(model.experiment.model_parameters)
        _p[model.bifurcation_index] = p 
        push!(params, _p)
    end
   
    model.roots = [roots(model.roots_function(p...), model.search_interval) for p in params]
end
