@kwdef struct PopulationExperimentSetup <: AbstractExperimentSetup
    init::Vector{Float64} 
    model_parameters::Vector{Float64} 
    tspan_simulation::Tuple{Float64, Float64} 
    tspan_analytical::Tuple{Float64, Float64} 
    Δt::Float64 
    max_pop::Int64 
    truncation::Vector{Int64} 
    jitt::Float64 = 1e-4
    molecular_model_rn::ReactionSystem
    division_rate
    effective_dilution_rn::ReactionSystem
end

function run_analytical(experiment::T; iters = 10, kwargs...) where T<:AbstractExperimentSetup
    model = CellPopulationModel(
        experiment.molecular_model_rn, 
        experiment.division_rate, 
        BinomialKernel(0.5))

    approximation = FiniteStateApprox(experiment.truncation, experiment.tspan_analytical)  
    solver = AnalyticalSolverParameters(experiment.truncation, iters; kwargs...)

    return solvecme(model, experiment, approximation, solver)
end


@kwdef struct ParameterStudySetup <: AbstractExperimentSetup 
    bifurcation_index::Int64
    parameter_span::Tuple{Float64, Float64}
    simulation_pstep::Float64
    effective_pstep::Float64 
    parametrisations::Vector{Vector{Float64}} 
    save_parameters::Vector{Float64} # Only use the first parameter in the parametrisations for saving. 
    note::String

    function ParameterStudySetup(;
        bifurcation_index, 
        parameter_span, 
        simulation_pstep,
        effective_pstep, 
        parameter_vector, 
        note) 
        
        parametrisations = [begin vec = deepcopy(parameter_vector); vec[bifurcation_index] = v; vec end for v in 
            range(parameter_span[1], step = simulation_pstep, stop = parameter_span[2])]

        new(bifurcation_index, 
           parameter_span, 
           simulation_pstep, 
           effective_pstep, 
           parametrisations, 
           parametrisations[1],
           note)
    end
end

function simulation_prange(setup::ParameterStudySetup)
    return range(setup.parameter_span[1], step = setup.simulation_pstep, stop = setup.parameter_span[2])
end

function effective_prange(setup::ParameterStudySetup)
    return range(setup.parameter_span[1], step = setup.effective_pstep, stop = setup.parameter_span[2])
end


