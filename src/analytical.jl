struct AnalyticalModel
    experiment::AbstractExperimentSetup
    cme_model::Function
    partition_kernel::Function
    division_rate::Function
end

struct AnalyticalSolverParameters
    truncation::Vector{Int64}
    maxiters::Int64
    solver::Any
    rtol::Float64
    atol::Float64
    rootfinder::Roots.AbstractSecant

    function AnalyticalSolverParameters(truncation, maxiters;solver=Vern7(), rootfinder=Order1(), rtol=1e-6, atol=1e-6)
        new(truncation, maxiters, solver, rtol, atol, rootfinder)
    end
end

mutable struct ConvergenceMonitor
    # TODO: Monitor convergence of the results. Here we want within the
    # iteration. Error control wrt truncation separately.
    birth_dists::Array{Vector{Float64}}
    growth_factor::Array{Float64}

    function ConvergenceMonitor()
        new([], [])
    end
end

mutable struct AnalyticalResults
    # TODO: do I extend AnalyticalResults every time I need a new statistic...
    # Or have a dictionary which can be extended freely?
    birth_dist::Vector{Float64}
    division_dist::Vector{Float64}
    growth_factor::Float64 
    cme_solution
    model::AnalyticalModel
    solver::AnalyticalSolverParameters
    convergence_monitor::ConvergenceMonitor
    marginal_size::Vector{Float64}

    function AnalyticalResults()
        new()
    end
end

function Base.show(::IO, ::AnalyticalResults)
    println("Analytical computation results for cell population model.")
end

function marginal_size_distribution!(result::AnalyticalResults; rtol=1e-6, atol=1e-6)
    println("Calculating marginal size ...")
    λ = result.growth_factor
    Π(τ) = λ *
        hquadrature(s -> division_time_dist(result)(s), τ, result.model.experiment.tspan_analytical[end]; abstol=atol)[1]
    
    telaps = @elapsed marginalΠ = hquadrature(
        length(result.birth_dist), 
        (s,v) -> v[:] = result.cme_solution(s) * Π(s), 
        result.model.experiment.tspan_analytical[1], 
        result.model.experiment.tspan_analytical[end]; abstol=atol)[1]

    result.marginal_size = marginalΠ ./ sum(marginalΠ)
    println("Integration complete, took $telaps seconds.")
end

function mean_marginal_size(result::AnalyticalResults;)
    return sum(result.marginal_size .* collect(0:length(result.marginal_size)-1)) 
end

function first_passage_time(x::Union{Vector{Float64}, Vector{Int64}}, 
    τ::Real, p::Vector{Float64}, Π; results)
    # TODO: in general need to pass in also the reaction network to see which
    # indices correspond to which counts. 
    model = results.model

    states = CartesianIndices(zeros(results.solver.truncation...))
    states = map(x -> x.I .- tuple(I), states)
    
    # First passage time for division.
    return model.division_rate.(states, fill(p, size(states)), τ) .* Π(τ) 
end

function division_time_dist(results::AnalyticalResults)
    model = results.model
    experiment = results.model.experiment
    return τ -> sum(
        first_passage_time(results.birth_dist, τ, experiment.model_parameters, results.cme_solution; results=results))
end

function division_dist(results::AnalyticalResults) 
    model = results.model
    experiment = results.model.experiment
    return hquadrature(
        length(results.birth_dist),
        (s,v) -> v[:] = first_passage_time(
            results.birth_dist, 
            s, 
            experiment.model_parameters, 
            results.cme_solution; 
            results=results), 
        experiment.tspan_analytical[1], 
        experiment.tspan_analytical[2], 
        reltol=results.solver.rtol, 
        abstol=results.solver.atol)[1]
end

function division_dist_hist(results::AnalyticalResults) 
    model = results.model
    experiment = results.model.experiment
    return hquadrature(
        length(results.birth_dist),
        (s,v) -> v[:] = 2 * 
            exp(-results.growth_factor * s) * 
            first_passage_time(
                results.birth_dist, 
                s, 
                experiment.model_parameters, 
                results.cme_solution; 
                results=results), 
        experiment.tspan_analytical[1], 
        experiment.tspan_analytical[2], 
        reltol=results.solver.rtol, 
        abstol=results.solver.atol)[1]
end

function update_growth_factor!(model::AnalyticalModel, results::AnalyticalResults, 
    solver::AnalyticalSolverParameters, cme_solution::ODESolution)

    experiment = results.model.experiment
    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 
    marginalfpt(τ) = sum(
        first_passage_time(results.birth_dist, τ, experiment.model_parameters, cme_solution; results=results))

    # Root finding for the population growth rate λ.
    f(λ) = 1 - 2 *  hquadrature(
        s -> marginalfpt(s) * exp(-λ*s), 
        experiment.tspan_analytical[1], 
        experiment.tspan_analytical[2], 
        reltol=solver.rtol, 
        abstol=results.solver.atol)[1]

    results.growth_factor = find_zero(f, 0, solver.rootfinder)
end

function update_birth_dist!(model::AnalyticalModel, results::AnalyticalResults, 
    solver::AnalyticalSolverParameters, cme_solution::ODESolution)

    experiment = results.model.experiment

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand(τ) = sum(
        model.partition_kernel.(
            collect.(axes(results.birth_dist))[1] .- 1, 
            solver.truncation) 
        .* first_passage_time(
            results.birth_dist, 
            τ, 
            experiment.model_parameters, 
            cme_solution; 
            results=results) 
        .* 2 .*exp(-results.growth_factor*τ))

    results.birth_dist = hquadrature(
        length(results.birth_dist),
        (s,v) -> v[:] = boundary_cond_integrand(s), 
        experiment.tspan_analytical[1], 
        experiment.tspan_analytical[2], 
        reltol=results.solver.rtol,
        abstol=results.solver.atol)[1]

    results.birth_dist = results.birth_dist / sum(results.birth_dist)
end

function log_convergece!(convergence::ConvergenceMonitor, results::AnalyticalResults)
    push!(convergence.birth_dists, results.birth_dist)
    push!(convergence.growth_factor, results.growth_factor)
end

function random_initial_values(model::AnalyticalModel)
    truncation = prod(model.experiment.truncation)
    # Random normalised initial conditions.
    init = rand(1:truncation, truncation) 
    return  init ./ sum(init)
end

function solvecme(model::AnalyticalModel,
    solver::AnalyticalSolverParameters)

    experiment = model.experiment

    # Every interation refines birth_dist (Π(x|0)) and growth factor λ.
    results = AnalyticalResults()
    results.model = model
    results.solver = solver

    convergence = ConvergenceMonitor()
    results.birth_dist = random_initial_values(model)
    i::Int64 = 0

    progress = Progress(solver.maxiters;)
     
    while i < solver.maxiters
        # Solve the CME Π(x|τ).
        cme = model.cme_model(results.birth_dist, experiment.tspan_analytical, experiment.model_parameters, solver.truncation)
        cme_solution = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 

        update_growth_factor!(model, results, solver, cme_solution)
        update_birth_dist!(model, results, solver, cme_solution)
        log_convergece!(convergence, results)

        i += 1
        ProgressMeter.next!(progress, showvalues = [("Current iteration", i), ("Growth factor λₙ", results.growth_factor)])
    end

    cme = model.cme_model(results.birth_dist, experiment.tspan_analytical, experiment.model_parameters, solver.truncation)
    cme_solution = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 
    results.cme_solution = cme_solution
    results.convergence_monitor = convergence

    return results 
end
