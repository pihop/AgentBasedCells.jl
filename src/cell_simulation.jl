struct SimulationSolver
    Δt::Float64
    jitt::Float64
    max_pop::Int64
end

mutable struct CellState
    # Keeps track of the cell state and cell age.
    state::Array{Float64}
    τ::Float64 
    birth_time::Float64
    birth_state::Array{Float64}
    division_time::Float64
    division_sampler::NonHomogeneousSampling
end

function partition_cell(cell::CellState, birth_time::Float64)
    # Parition and return the Array of cells.    
    partition = [] 
    partition_ = [] 
    for s in cell.state
        molecule_number = rand(Binomial(Int(s), 0.5))
        push!(partition, molecule_number)
        push!(partition_, Int(s) - molecule_number)  
    end

    return [CellState(partition, 0.0, birth_time, partition, 0.0, ThinningSampler()), 
            CellState(partition_, 0.0, birth_time, partition_, 0.0, ThinningSampler())]
end

function apply_chemostat!(cell_population::Array{CellState}, pop_size_max::Int64)
    if length(cell_population) > pop_size_max
        splice!(cell_population, sort(sample(1:length(cell_population), length(cell_population) - pop_size_max; replace=false)))
    end
end

mutable struct CellSimulationProblem <: AbstractSimulationProblem 
    model::CellPopulationModel
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
        new(model, jump, init, ps, tspan)
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
            :final_population => [])

        new(dict, model, solver)
    end
end

function Base.show(io::IO, ::CellSimulationResults)
    print(io, "Direct stochastic simulations results for cell population model.")
end

function final_cell_sizes(results::CellSimulationResults)
    return [cell.state for cell in results.final_population]
end

function molecule_state(cell_state::CellState)
    return cell_state.state
end

function birth_molecule_state(cell_state::CellState)
    return cell_state.birth_state
end

function cell_size(cell_state::CellState)
    return sum(cell_state.state)
end

function cell_birth_size(cell_state::CellState)
    return sum(cell_state.birth_state)
end

function cell_age(cell_state::CellState)
    return cell_state.τ
end

function cell_division_time(cell_state::CellState)
    return cell_state.division_time
end

function cell_birth_time(cell_state::CellState)
    return cell_state.birth_time
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

function compute_division_times!(problem::CellSimulationProblem, solver::SimulationSolver, cell_population::Array{CellState})
    # Simulate all cells up to next division.
    for (index, cell) in collect(enumerate(cell_population))
        while true
            sim = simulate_molecular(problem.molecular_model, 
                cell, problem.ps, (cell.τ, cell.τ + solver.Δt + solver.jitt);)
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
end

function states_at_last_division!(problem, results, division_time, cell_population; jitt)
    for cell in cell_population
        sim = simulate_molecular(problem.molecular_model, cell, problem.ps,
            (0.0, division_time - cell.birth_time + jitt); from_birth=true)
         
        push!(results.results[:final_population], sim.u[end])
    end
end

function simulate(problem::CellSimulationProblem, solver::SimulationSolver)
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

        daughter_cells = partition_cell(cell_population[cell_idx], next_division_time)
        compute_division_times!(problem, solver, daughter_cells)
        log_results!(simulation_results, cell_population[cell_idx], daughter_cells)

        # Remove the mother cell and replace with daughters.
        deleteat!(cell_population, cell_idx)

        push!(cell_population, daughter_cells...)

        t = next_division_time
        ProgressMeter.next!(progress, showvalues = [("Time", t), ("Population size", length(cell_population))])
    end

    states_at_last_division!(problem, simulation_results, t, cell_population; solver.jitt)
    return simulation_results 
end
