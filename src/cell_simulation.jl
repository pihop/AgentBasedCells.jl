mutable struct CellState
    # Keeps track of the cell state and cell age.
    state::Array{Float64}
    τ::Float64 
    birthTime::Float64
    divisionSampler::NonHomogeneousSampling
end

mutable struct CellSimulationResults
    t::Vector{Float64}
    populationSize::Vector{Int64}
    divisionTimes::Vector{Float64}
    moleculesAtDivision::Vector{Int64}
    moleculesAtBirth::Vector{Int64}

    function CellSimulationResults()
        new([], [], [], [], [])
    end
end

function cellSize(cellState::CellState)
    return sum(cellState.state)
end

function simulateMolecular(molecularModel::JumpProblem,
    CellState::CellState, 
    modelParameters::Vector{Float64}, 
    tspan::Tuple{Float64, Float64})
    # TODO: We can use multiple dispatch to define versions for ODEProblem,
    # SDEProblem or other for molecularModel.
    
    jprob = remake(molecularModel, u0=CellState.state, p=modelParameters, tspan=tspan)
    return solve(jprob, SSAStepper())
end

function simulatePopulation(CellPopulation::Array{CellState}, 
        MolecularModel::Union{JumpProblem, ODEProblem, SDEProblem}, 
        DivisionModel::Function,
        partitionCell::Function,
        chemostatFunction::Function,
        modelParameters::Vector{Float64}, 
        tspan::Tuple{Float64, Float64}, 
        Δt::Float64, 
        maxN::Int64; 
        jitt::Float64)

    # Progress monitor.
    t::Float64 = tspan[1]        
    simulationResults = CellSimulationResults()
    progress = Progress(Int((tspan[end]-tspan[1])/Δt);)

    while t < tspan[end] 
        CellPopulationWorking = deepcopy(CellPopulation)
        NewCellPopulation::Array{CellState} = []

        while !isempty(CellPopulationWorking)
            NewCellPopulationWorking::Array{CellState} = []
            for (index, Cell) in collect(enumerate(CellPopulationWorking))
                simTime = t + Δt - Cell.birthTime - Cell.τ   
                sim = simulateMolecular(MolecularModel, Cell, modelParameters, (Cell.τ, Cell.τ + simTime + jitt))
                division_time = DivisionModel(sim, Cell.divisionSampler, (Cell.τ, Cell.τ + simTime))

                if division_time == nothing
                    Cell.state = sim(Cell.τ + simTime)
                    Cell.τ = Cell.τ + simTime 
                    push!(NewCellPopulation, Cell)
                else
                    # Division to daughter cells. 
                    daughterCells = partitionCell(Cell, t + division_time - Cell.τ)

                    # Log the results.
                    push!(simulationResults.divisionTimes, division_time)
                    push!(simulationResults.moleculesAtBirth, cellSize.(daughterCells)...)
                    push!(simulationResults.moleculesAtDivision, cellSize(CellPopulationWorking[index]))

                    # Remove the mother cell and replace with daughters.
                    push!(NewCellPopulationWorking, daughterCells...)
                end
            end
            CellPopulationWorking = deepcopy(NewCellPopulationWorking)
        end

        CellPopulation = deepcopy(NewCellPopulation)
        # chemostat (keep constant cell population)
        # NOTE!!! If simulation time step Δt is too large the working population
        # might explode before chemostat is called.
        chemostatFunction(CellPopulation, maxN)
       
        t += Δt
        ProgressMeter.next!(progress, showvalues = [("Time", t), ("Population size", length(CellPopulation))])
    end

    return simulationResults
end



