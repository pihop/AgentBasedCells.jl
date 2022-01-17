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

mutable struct CellSimulationModel
    xinit::Vector{Float64}
    model_parameters::Vector{Float64}
    cell_population::Array{CellState}
    molecular_model::Union{JumpProblem, ODEProblem, SDEProblem}
    division_model::Function
end

struct CellSimulationParameters
    tspan::Tuple{Float64, Float64}
    Δt::Float64
    max_pop::Int64
    jitt::Float64

    function CellSimulationParameters(tspan, Δt, max_pop, jitt)
        new(tspan, Δt, max_pop, jitt) 
    end
end

mutable struct CellSimulationResults
    t::Vector{Float64}
    population_size::Vector{Int64}
    division_times_all::Vector{Float64} # Final generation.
    division_times_ancest::Vector{Float64} # Ancestry.
    molecules_at_division::Vector{Int64}
    molecules_at_division_all::Vector{Int64}
    molecules_at_birth::Vector{Int64}
    molecules_at_birth_all::Vector{Int64}
    final_population::Array{Vector{Float64}}

    function CellSimulationResults()
        new([], [], [], [], [], [], [], [], [])
    end
end

function Base.convert(::Type{Dict}, results::CellSimulationResults)
    return Dict(string.(fieldnames(CellSimulationResults)) .=> getfield.(Ref(results), fieldnames(CellSimulationResults)))
end

function final_cell_sizes(results::CellSimulationResults)
    return [cell.state for cell in results.final_population]
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

function simulate_molecular(molecular_model::JumpProblem,
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

function simulate_molecular_from_birth(molecular_model::JumpProblem,
    cell_state::CellState,
    model_parameters::Vector{Float64},
    tspan::Tuple{Float64, Float64})

    jprob = remake(molecular_model, u0=cell_state.birth_state, p=model_parameters, tspan=tspan)
    return solve(jprob, SSAStepper())
end

function compute_division_times!(model::CellSimulationModel, simulation_params::CellSimulationParameters, cell_population::Array{CellState})
    # Simulate all cells up to next division.
   
    for (index, cell) in collect(enumerate(cell_population))
        while true
            sim = simulate_molecular(model.molecular_model, 
                cell, model.model_parameters, (cell.τ, cell.τ + simulation_params.Δt + simulation_params.jitt);)
    
            division_time = model.division_model(sim, 
                model.model_parameters, cell.division_sampler, (cell.τ, cell.τ + simulation_params.Δt))
    
            if division_time == nothing
                cell.state = sim(cell.τ + simulation_params.Δt)
                cell.τ = cell.τ + simulation_params.Δt 
            else
                cell.state = sim(division_time)
                cell.τ = division_time 
                cell.division_time = cell.birth_time + division_time 
                break
            end
        end
    end
end

function log_results!(simulation_results, mother, daughters)
    push!(simulation_results.division_times_ancest, mother.division_time - mother.birth_time)
    push!(simulation_results.molecules_at_division, cell_size(mother))
    push!(simulation_results.molecules_at_birth, cell_birth_size(mother))
    
    push!(simulation_results.division_times_all, (cell_division_time.(daughters) .- cell_birth_time.(daughters))...)
    push!(simulation_results.molecules_at_birth_all, cell_birth_size.(daughters)...)
    push!(simulation_results.molecules_at_division_all, cell_size.(daughters)...)
end

function states_at_last_division!(model, simulation_results, division_time, cell_population)
    for cell in cell_population
        sim = simulate_molecular(model.molecular_model, cell, model.model_parameters,
            (0.0, division_time - cell.birth_time); from_birth=true)
         
        push!(simulation_results.final_population, sim.u[end])
    end
end

function simulate_population(model::CellSimulationModel,
        simulation_params::CellSimulationParameters)

    # Progress monitor.
    t::Float64 = simulation_params.tspan[1]        
    simulation_results = CellSimulationResults()
    progress = ProgressUnknown()
    # Initial cell population.
    cell_population = model.cell_population
    compute_division_times!(model, simulation_params, cell_population)

    while t < simulation_params.tspan[end]
        apply_chemostat!(cell_population, simulation_params.max_pop)
        next_division_time, cell_idx = findmin(cell_division_time.(cell_population))

        daughter_cells = partition_cell(cell_population[cell_idx], next_division_time)
        compute_division_times!(model, simulation_params, daughter_cells)
        log_results!(simulation_results, cell_population[cell_idx], daughter_cells)

        # Remove the mother cell and replace with daughters.
        deleteat!(cell_population, cell_idx)

        push!(cell_population, daughter_cells...)

        t = next_division_time
        ProgressMeter.next!(progress, showvalues = [("Time", t), ("Population size", length(cell_population))])
    end

    states_at_last_division!(model, simulation_results, t, cell_population)
    return simulation_results 
end

function simulate_population_slow(model::CellSimulationModel,
        simulation_params::CellSimulationParameters
    )

    # Progress monitor.
    t::Float64 = simulation_params.tspan[1]        
    simulation_results = CellSimulationResults()
    progress = Progress(Int((simulation_params.tspan[end]-simulation_params.tspan[1])/simulation_params.Δt);)
    # Initial cell population.
    cell_population = model.cell_population

    while t < simulation_params.tspan[end] 
        apply_chemostat!(cell_population, simulation_params.max_pop)
        _cell_population = deepcopy(cell_population)
        new_cell_population::Array{CellState} = []

        while !isempty(_cell_population)
            _new_cell_population::Array{CellState} = []
            for (index, cell) in collect(enumerate(_cell_population))

                sim_time = t + simulation_params.Δt - cell.birth_time - cell.τ   
                sim = simulate_molecular(model.molecular_model, 
                    cell, model.model_parameters, (cell.τ, cell.τ + sim_time + simulation_params.jitt))

                division_time = model.division_model(sim, 
                    model.model_parameters, cell.division_sampler, (cell.τ, cell.τ + sim_time))

                if division_time == nothing
                    cell.state = sim(cell.τ + sim_time)
                    cell.τ = cell.τ + sim_time 
                    push!(new_cell_population, cell)
                else
                    # Division to daughter cells. 
                    daughter_cells = partition_cell(cell, t + division_time - cell.τ)

                    # Remove the mother cell and replace with daughters.
                    push!(_new_cell_population, daughter_cells...)

                    log_results!(simulation_results, _cell_population[index], daughter_cells,  division_time)
                end
            end
            _cell_population = deepcopy(_new_cell_population)
        end

        cell_population = deepcopy(new_cell_population)
#        last_generation(model, cell_population, simulation_params, simulation_results)
        # chemostat (keep constant cell population)
        # NOTE!!! If simulation time step Δt is too large the working population
        # might explode before chemostat is called.
             
        t += simulation_params.Δt
        ProgressMeter.next!(progress, showvalues = [("Time", t), ("Population size", length(cell_population))])
    end

    return simulation_results
end


