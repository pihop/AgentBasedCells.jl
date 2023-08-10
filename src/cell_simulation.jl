struct SimulationSolver
    Δt::Float64
    jitt::Float64
    max_pop::Int64
end

mutable struct CellSimulationProblem{T} <: AbstractSimulationProblem 
    model::T
    molecular_model::Union{JumpProcesses.JumpProblem, ODEProblem, SDEProblem}
    init::Vector{CellState}
    ps::Vector{Float64}
    tspan::Tuple{Float64, Float64}

    function CellSimulationProblem(model, init, ps, tspan) 
        # Initial things.
        discrete = DiscreteProblem(
            model.molecular_model, 
            species(model.molecular_model) .=> init[1].state, 
            tspan, 
            Catalyst.parameters(model.molecular_model) .=> ps)

        jump = JumpProcesses.JumpProblem(model.molecular_model, discrete, Direct())
        new{typeof(model)}(model, jump, init, ps, tspan)
    end
end 

function apply_chemostat!(cell_population::Array{CellState}, pop_size_max::Int64)
    if length(cell_population) > pop_size_max
        splice!(cell_population, sort(sample(1:length(cell_population), length(cell_population) - pop_size_max; replace=false)))
    end
end

mutable struct CellSimulationResults
    results::Dict
    problem::CellSimulationProblem
    solver::SimulationSolver

    function CellSimulationResults(model::CellSimulationProblem, solver::SimulationSolver)
        dict = Dict(
            :division_times_ancest => [],
            :division_times_all => [],
            :molecules_at_division => [],
            :molecules_at_division_all => [],
            :molecules_at_birth => [],
            :molecules_at_birth_all => [],
            :all_population => [],
            :final_population => [],
            :final_population_traj => [],
            :trajectories => [],
            :pop_size => []
           )

        new(dict, model, solver)
    end
end

function Base.show(io::IO, ::CellSimulationResults)
    print(io, "Direct stochastic simulations results for cell population model.")
end

function final_cell_sizes(results::CellSimulationResults)
    return [cell.state for cell in results.final_population]
end

function simulate_molecular(
    molecular_model::JumpProcesses.JumpProblem,
    cell_state::CellState, 
    model_parameters::Vector{Float64}, 
    tspan::Tuple{Float64, Float64};
    from_birth::Bool=false)
    # TODO: We can use multiple dispatch to define versions for ODEProblem,
    # SDEProblem or other for molecularModel.
    
    if from_birth == false
        jprob = remake(molecular_model, u0=cell_state.state, p=model_parameters, tspan=tspan)
        return solve(jprob, SSAStepper())
    elseif from_birth == true
        jprob = remake(molecular_model, u0=cell_state.birth_state, p=model_parameters, tspan=tspan)
        return solve(jprob, SSAStepper())
    end
end

function append_sim!(cell, cellsim::Nothing, sim)
    cell.sim = sim
end

function append_sim!(cell, cellsim::ODESolution, sim)
    cell.sim = DiffEqBase.build_solution(sim.prob, cellsim.alg, vcat(cellsim.t, sim.t), vcat(cellsim.u, sim.u), successful_retcode=true)
end

function compute_division_times!(problem::CellSimulationProblem, solver::SimulationSolver, cell_population::Array{CellState})
    # Simulate all cells up to next division.
    for (index, cell) in collect(enumerate(cell_population))
        while true
            sim = simulate_molecular(problem.molecular_model, 
                cell, problem.ps, (cell.τ, cell.τ + solver.Δt);)
            append_sim!(cell, cell.sim, sim)
            division_time = sample_next_division(sim, (cell.τ, cell.τ + solver.Δt), problem, cell.division_sampler)
    
            if division_time == nothing
                cell.state = sim(cell.τ + solver.Δt)
                cell.τ = cell.τ + solver.Δt 
            else
                cell.state = sim(division_time)
                cell.τ = division_time 
                cell.division_time = cell.birth_time + division_time 
                break
            end
        end
    end
end

function log_results!(results, mother, daughters)
    push!(results.results[:division_times_ancest], mother.division_time - mother.birth_time)
    push!(results.results[:molecules_at_division], mother.state)
    push!(results.results[:molecules_at_birth], mother.birth_state)
    
    push!(results.results[:division_times_all], (cell_division_time.(daughters) .- cell_birth_time.(daughters))...)
    push!(results.results[:molecules_at_birth_all], birth_molecule_state.(daughters)...)
    push!(results.results[:molecules_at_division_all], molecule_state.(daughters)...)
    push!(results.results[:all_population], daughters...)
    push!(results.results[:trajectories], mother)
end

function log_pop_size!(results, t, cell_population)
    push!(results.results[:pop_size], (t, length(cell_population)))
end

function states_at_last_division!(problem, results, division_time, cell_population; jitt)
#    for cell in cell_population
#        display(cell.sim.t[end])
#        display(division_time - cell.birth_time)
#        sim = simulate_molecular(problem.molecular_model, cell, problem.ps,
#            (cell.sim.t[end], division_time - cell.birth_time); from_birth=false)
#         
#        append_sim!(cell, cell.sim, sim)
#        push!(results.results[:final_population], sim.u[end])
#    end
    results.results[:final_population_traj] = cell_population
end

function simulate(problem::CellSimulationProblem{T}, solver::SimulationSolver) where T <: Union{CellPopulationModel, MotherCellModel}
    # Progress monitor.
    t = problem.tspan[1]
    simulation_results = CellSimulationResults(problem, solver)
    progress = ProgressUnknown()
    # Initial cell population.
    cell_population = problem.init
    compute_division_times!(problem, solver, cell_population)

    while t < problem.tspan[end]
        apply_chemostat!(cell_population, solver.max_pop)
        next_division_time, cell_idx = findmin(cell_division_time.(cell_population))

        daughter_cells = partition_cell(
            T,
            problem.model.partition_kernel, 
            cell_population[cell_idx], 
            next_division_time)

        compute_division_times!(problem, solver, daughter_cells)
        log_results!(simulation_results, cell_population[cell_idx], daughter_cells)

        # Remove the mother cell and replace with daughters.
        deleteat!(cell_population, cell_idx)

        if floor(t / solver.Δt) < floor(next_division_time / solver.Δt)
            log_pop_size!(simulation_results, next_division_time, cell_population)
        end

        push!(cell_population, daughter_cells...)

        t = next_division_time
        ProgressMeter.next!(progress, showvalues = [("Time", t), ("Population size", length(cell_population))])
    end
    states_at_last_division!(problem, simulation_results, t, cell_population; solver.jitt)
    return simulation_results 
end
