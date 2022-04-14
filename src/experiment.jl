@kwdef struct PopulationExperimentSetup <: AbstractExperimentSetup
    init::Vector{Float64} 
    ps::Vector{Float64} 
    simulation_tspan::Tuple{Float64, Float64} 
    analytical_tspan::Tuple{Float64, Float64} 
    iters::Int64
    Δt::Float64 
    max_pop::Int64 
    truncation::Vector{Int64} 
    jitt::Float64 = 1e-4
end

function run_analytical(model, exp::T; kwargs...) where T <: AbstractExperimentSetup
    approx = FiniteStateApprox(exp.truncation)
    problem = AnalyticalProblem(model, exp.init, exp.ps, exp.analytical_tspan, approx) 
    solver = AnalyticalSolver(exp.iters)

    return solvecme(problem, solver)
end

_run_analytical(model, exp) = run_analytical(model, exp)
Broadcast.broadcasted(::typeof(run_analytical), model, exp) = 
    broadcast(_run_analytical, Ref(model), exp)

function run_simulation(model, exp::T) where T <: AbstractExperimentSetup
    init_pop = [CellState([0, ], 0.0, 0.0, [0.0, ], 0.0, CellPopulationSimulations.ThinningSampler()), ]
    problem = CellSimulationProblem(model, init_pop, exp.ps, exp.simulation_tspan)  
    solver = SimulationSolver(exp.Δt, exp.jitt, exp.max_pop) 

    return simulate(probem, solver)
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
