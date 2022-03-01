module CellPopulationSimulations

using Base: @kwdef
using ProgressMeter
using LinearAlgebra
using RuntimeGeneratedFunctions

using DiffEqBase
using DiffEqJump
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqCallbacks
using ModelingToolkit
using Catalyst
using Distances

using Cubature

using IntervalArithmetic: Interval
using IntervalRootFinding
using Roots

abstract type NonHomogeneousSampling end
abstract type AbstractSimulationModel end
abstract type AbstractPartitionKernel end
abstract type AbstractExperimentSetup end
abstract type AbstractPopulationModel end
abstract type AbstractAnalyticalApprox end

include("partition_kernels.jl")
export BinomialKernel 

include("model.jl")

using Distributions
include("thinning.jl")

using FiniteStateProjection
using SparseArrays
include("analytical_approximations.jl")

include("cell_simulation.jl")
include("analytical.jl")

# Utils for the symbolically defined aspects of the model. 
using Symbolics: value
using SymbolicUtils
include("symbolics.jl")

include("effective_dilution.jl")
include("stochastic_dilution.jl")

include("experiment.jl")
export AbstractExperimentSetup, run_analytical, run_simulation

export AnalyticalModel, AnalyticalResults, AnalyticalSolverParameters
export marginal_size_distribution!, mean_marginal_size
export growth_factor, division_dist, division_dist_hist, division_time_dist, division_time_dist_hist
#export ThinningSampler, sample_first_arrival!
export CellState, CellSimulationResults, CellSimulationParameters, CellSimulationModel
export cellsize, final_cell_sizes
export simulate_population, simulate_population_slow
#export solvecme
#export gen_division_rate_function

export EffectiveDilutionModel
export root_finding!
export Interval

export PopulationExperimentSetup

export ParameterStudySetup
export simulation_prange, effective_prange

export StochasticDilutionModel, birth_death_ss!, mean_steady_state

#export BurstyReactionModel

end # module
