# Interface we need from the experiment setup.

function parameters(::AbstractExperimentSetup) end
function init_values(::AbstractExperimentSetup) end
function simulation_tspan(::AbstractExperimentSetup) end
function analytical_tspan(::AbstractExperimentSetup) end
function Δt(::AbstractExperimentSetup) end
function truncation(::AbstractExperimentSetup) end
function jitt(::AbstractExperimentSetup) end
function molecular_model(::AbstractExperimentSetup) end
function symbolic_division_rate(::AbstractExperimentSetup) end

function run_analytical(experiment::T; iters = 10, kwargs...) where T<:AbstractExperimentSetup
    model = CellPopulationModel(
        molecular_model(experiment), 
        symbolic_division_rate(experiment), 
        BinomialKernel(0.5))

    approximation = FiniteStateApprox(truncation(experiment), analytical_tspan(experiment))  
    solver = AnalyticalSolverParameters(truncation(experiment), iters; kwargs...)

    return solvecme(model, experiment, approximation, solver)
end

function run_simulation(experiment::T) where T<:AbstractExperimentSetup
    model = CellPopulationModel(
        molecular_model(experiment), 
        symbolic_division_rate(experiment), 
        BinomialKernel(0.5))

    initial_population = [CellState([0, ], 0.0, 0.0, [0.0, ], 0.0, ThinningSampler()), ]
    simulation_model = CellSimulationModel(model, experiment, initial_population)
    results = simulate_population(simulation_model)

    sname = savename(experiment; allowedtypes = (Vector, String, Int, Float64))
    save(datadir("sims", "cell_simulation", "$sname.jld2"), convert(Dict, results))
    return results
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

@kwdef struct PopulationExperimentSetup <: AbstractExperimentSetup
    init::Vector{Float64} 
    model_parameters::Vector{Float64} 
    simulation_tspan::Tuple{Float64, Float64} 
    analytical_tspan::Tuple{Float64, Float64} 
    Δt::Float64 
    max_pop::Int64 
    truncation::Vector{Int64} 
    jitt::Float64 = 1e-4
    molecular_model::ReactionSystem
    division_rate
    effective_dilution::ReactionSystem
end

parameters(setup::PopulationExperimentSetup) = setup.model_parameters
init_values(setup::PopulationExperimentSetup) = setup.init
simulation_tspan(setup::PopulationExperimentSetup) = setup.simulation_tspan
analytical_tspan(setup::PopulationExperimentSetup) = setup.analytical_tspan
Δt(setup::PopulationExperimentSetup) = setup.Δt
truncation(setup::PopulationExperimentSetup) = setup.truncation
jitt(setup::PopulationExperimentSetup) = setup.jitt
molecular_model(setup::PopulationExperimentSetup) = setup.molecular_model
symbolic_division_rate(setup::PopulationExperimentSetup) = setup.division_rate

