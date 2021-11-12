module CellPopulationSimulations

using Distributions
using ProgressMeter
using DiffEqJump
using OrdinaryDiffEq
using StochasticDiffEq

abstract type NonHomogeneousSampling end

include("thinning.jl")
include("cell_simulation.jl")

export ThinningSampler, sampleFirstArrival!
export CellState, CellSimulationResults
export cellSize
export simulatePopulation

end # module
