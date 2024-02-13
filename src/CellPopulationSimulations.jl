module CellPopulationSimulations

using Base: @kwdef
using ProgressMeter
using LinearAlgebra
using RuntimeGeneratedFunctions

using DiffEqBase
using SciMLBase
using JumpProcesses
import OrdinaryDiffEq: Rodas5

using StochasticDiffEq
using DiffEqCallbacks
import Catalyst: ReactionSystem
import Distances: totalvariation
using PaddedViews
using PreallocationTools

using Integrals
using Cubature
using FastGaussQuadrature

using Roots

import ArraysOfArrays: ArrayOfSimilarArrays
using PreallocationTools
using LoopVectorization

import FunctionWrappers: FunctionWrapper

abstract type NonHomogeneousSampling end
abstract type AbstractSimulationProblem end
abstract type AbstractPartitionKernel end
abstract type AbstractExperimentSetup end
abstract type AbstractPopulationModel end
abstract type AbstractAnalyticalApprox end

include("cellstate.jl")
export CellState
export cell_index
export state_at_division, state_at_times, lineage, state_time, division_time_state

include("model.jl")
export CellPopulationModel, MotherCellModel, DivisionRateMonotonicInc, DivisionRateBounded

include("partition_kernels.jl")
export BinomialKernel, BinomialWithDuplicate, ConcentrationKernel

using Distributions
include("thinning.jl")

include("cell_simulation.jl")
export CellSimulationProblem, SimulationSolver
export simulate
include("analytical.jl")
export AnalyticalProblem, AnalyticalSolver, solvecme
export ToxicBoundaryRe, ToxicBoundaryDeath, Reinsert, Divide

using FiniteStateProjection
using SparseArrays
include("make_sparse_mat.jl")
include("analytical_approximations.jl")
export FiniteStateApprox 

# Utils for the symbolically defined aspects of the model. 
import Symbolics: value
using SymbolicUtils
include("symbolics.jl")

include("experiment.jl")
export AbstractExperimentSetup, run_analytical, run_analytical_single, run_analyticalerr_single, run_simulation, make_analytical_problem

export AnalyticalModel, AnalyticalResults, AnalyticalSolverParameters
export marginal_size_distribution!, mean_marginal_size, compute_joint!, marginal_age_distribution, marginal_gamma, division_hazard
export growth_factor, division_dist!, interdivision_time_dist, division_time_cdist!, joint_fpt_ancest_cdist!, division_dist_ancest!, division_time_dist, division_time_ancest, compute_growth_factor, fpt_dist_ancest
export birth_dist_hist!
export CellSimulationResults, CellSimulationParameters, CellSimulationProblem
export cellsize, final_cell_sizes
export simulate_population, simulate_population_slow
export cell_division_age

export Interval

export PopulationExperimentSetup

export ParameterStudySetup
export simulation_prange, effective_prange

end # module
