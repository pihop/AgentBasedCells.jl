mutable struct CellState
    # Keeps track of the cell state and cell age.
    state::Array{Float64}
    τ::Float64 
    birth_time::Float64
    division_sampler::NonHomogeneousSampling
end

mutable struct CellSimulationModel
    xinit::Vector{Float64}
    model_parameters::Vector{Float64}
    cell_population::Array{CellState}
    molecular_model::Union{JumpProblem, ODEProblem, SDEProblem}
    division_model::Function
    partition_cell::Function
    chemostat::Function
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
    division_times::Vector{Float64}
    division_times_all::Vector{Float64}
    molecules_at_division::Vector{Int64}
    molecules_at_birth::Vector{Int64}

    function CellSimulationResults()
        new([], [], [], [], [], [])
    end
end

function Base.convert(::Type{Dict}, results::CellSimulationResults)
    return Dict(string.(fieldnames(CellSimulationResults)) .=> getfield.(Ref(results), fieldnames(CellSimulationResults)))
end

function cell_size(cell_state::CellState)
    return sum(cell_state.state)
end

function simulate_molecular(molecular_model::JumpProblem,
    cell_state::CellState, 
    model_parameters::Vector{Float64}, 
    tspan::Tuple{Float64, Float64})
    # TODO: We can use multiple dispatch to define versions for ODEProblem,
    # SDEProblem or other for molecularModel.
    
    jprob = remake(molecular_model, u0=cell_state.state, p=model_parameters, tspan=tspan)
    return solve(jprob, SSAStepper())
end

function simulate_population(model::CellSimulationModel,
        simulation_params::CellSimulationParameters
    )

    # Progress monitor.
    t::Float64 = simulation_params.tspan[1]        
    simulation_results = CellSimulationResults()
    progress = Progress(Int((simulation_params.tspan[end]-simulation_params.tspan[1])/simulation_params.Δt);)
    # Initial cell population.
    cell_population = model.cell_population

    while t < simulation_params.tspan[end] 
       # display(length(cell_population))
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
                    daughter_cells = model.partition_cell(cell, t + division_time - cell.τ)

                    # Log the results.
                    push!(simulation_results.division_times, division_time)
                    push!(simulation_results.molecules_at_birth, cell_size.(daughter_cells)...)
                    push!(simulation_results.molecules_at_division, cell_size(_cell_population[index]))

                    # Remove the mother cell and replace with daughters.
                    push!(_new_cell_population, daughter_cells...)
                end
            end
            _cell_population = deepcopy(_new_cell_population)
        end
        cell_population = deepcopy(new_cell_population)
        # chemostat (keep constant cell population)
        # NOTE!!! If simulation time step Δt is too large the working population
        # might explode before chemostat is called.
        model.chemostat(cell_population, simulation_params.max_pop)
       
        t += simulation_params.Δt
        ProgressMeter.next!(progress, showvalues = [("Time", t), ("Population size", length(cell_population))])
    end

    return simulation_results
end



