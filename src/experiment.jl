@kwdef struct PopulationExperimentSetup <: AbstractExperimentSetup
    init::Vector{Float64} = Float64[]
    ps::Vector{Float64} = Float64[]
    simulation_tspan::Tuple{Float64, Float64} = (0.0, 0.0)
    analytical_tspan::Tuple{Float64, Float64} = (0.0, 0.0) 
    iters::Int64 = 0
    Δt::Float64 = 0.0
    max_pop::Int64 = 0
    truncation::Vector{Int64} = Int64[]
    jitt::Float64 = 1e-4
end

function run_analytical_single(model, exp::T; kwargs...) where T <: AbstractExperimentSetup
    approx = FiniteStateApprox(exp.truncation, model, exp.ps)
    problem = AnalyticalProblem(model, exp.ps, exp.analytical_tspan, approx) 
    solver = AnalyticalSolver(exp.iters; kwargs...)
    return solvecme(problem, solver)
end

function run_analyticalerr_single(model, exp::T; kwargs...) where T <: AbstractExperimentSetup
    approx = FiniteStateApprox(exp.truncation, model, exp.ps)
    problem = AnalyticalProblem(model, exp.ps, exp.analytical_tspan, approx) 
    solver = AnalyticalSolver(exp.iters; kwargs...)
    return solvecmeerror(problem, solver)
end

function run_analytical(model::CellPopulationModel, exp::Vector{T}; kwargs...) where T <: AbstractExperimentSetup
    return [run_analytical_single(model, e; kwargs...) for e in exp]
end

function run_analytical(model::MotherCellModel, exp::Vector{T}; kwargs...) where T <: AbstractExperimentSetup
    return [run_analytical_single(model, e; kwargs...) for e in exp]
end

_run_analytical(model, exp;) = run_analytical(model, exp;)
Broadcast.broadcasted(::typeof(run_analytical), model, exp;) = 
    broadcast(_run_analytical, Ref(model), exp;)

function run_simulation(model, exp::T) where T <: AbstractExperimentSetup
    init_pop = [CellState(exp.init, 0.0, 0.0, exp.init, 0.0, CellPopulationSimulations.ThinningSampler()), ]
    problem = CellSimulationProblem(model, init_pop, exp.ps, exp.simulation_tspan)  
    solver = SimulationSolver(exp.Δt, exp.jitt, exp.max_pop) 

    return simulate(problem, solver)
end

#@kwdef struct ParameterStudySetup <: AbstractExperimentSetup 
#    bifurcation_index::Int64
#    parameter_span::Tuple{Float64, Float64}
#    simulation_pstep::Float64
#    effective_pstep::Float64 
#    parametrisations::Vector{Vector{Float64}} 
#    save_parameters::Vector{Float64} # Only use the first parameter in the parametrisations for saving. 
#    note::String
#
#    function ParameterStudySetup(;
#        bifurcation_index, 
#        parameter_span, 
#        simulation_pstep,
#        effective_pstep, 
#        parameter_vector, 
#        note) 
#        
#        parametrisations = [begin vec = deepcopy(parameter_vector); vec[bifurcation_index] = v; vec end for v in 
#            range(parameter_span[1], step = simulation_pstep, stop = parameter_span[2])]
#
#        new(bifurcation_index, 
#           parameter_span, 
#           simulation_pstep, 
#           effective_pstep, 
#           parametrisations, 
#           parametrisations[1],
#           note)
#    end
#end
