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
    cmesol
    results::Dict 
    experiment::AbstractExperimentSetup
    approximation::AbstractAnalyticalApprox
    model::AbstractPopulationModel
    solver::AnalyticalSolverParameters
    convergence_monitor::ConvergenceMonitor

    function AnalyticalResults()
        results = new()
        results.results = Dict()
        return results
    end
end

function Base.show(::IO, ::AnalyticalResults)
    println("Analytical computation results for cell population model.")
end

function marginal_size_distribution!(results::AnalyticalResults; rtol=1e-6, atol=1e-6)
    println("Calculating marginal size ...")
    λ = results.results[:growth_factor]
    Π(τ) = λ *
        hquadrature(s -> division_time_dist(results)(s), τ, results.approximation.tspan[end]; abstol=atol)[1]
    
    telaps = @elapsed marginalΠ = hquadrature(
        length(results.results[:birth_dist]), 
        (s,v) -> v[:] = results.cmesol(s) * Π(s), 
        results.approximation.tspan[1], 
        results.approximation.tspan[end]; abstol=atol)[1]

    results.results[:marginal_size] = marginalΠ ./ sum(marginalΠ)
    println("Integration complete, took $telaps seconds.")
end

function mean_marginal_size(results::AnalyticalResults;)
    return sum(results.results[:marginal_size] .* collect(0:length(result.results[:marginal_size])-1)) 
end

function first_passage_time(
        x::Union{Vector{Float64}, Vector{Int64}}, 
        τ::Real, 
        p::Vector{Float64}, 
        Π; 
        results)

    # TODO: in general need to pass in also the reaction network to see which
    # indices correspond to which counts. 

    states = CartesianIndices(zeros(results.solver.truncation...))
    states = map(x -> x.I .- tuple(I), states)

    # First passage time for division.
    return results.model.division_rate.(states, fill(p, size(states)), τ) .* Π(τ) 
end

function division_time_dist(results::AnalyticalResults)
    experiment = results.experiment
    return τ -> sum(
        first_passage_time(
            results.results[:birth_dist], 
            τ, 
            experiment.model_parameters, 
            results.cmesol; 
            results=results))
end

function division_dist(results::AnalyticalResults) 
    experiment = results.experiment
    approximation = results.approximation

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
        approximation.tspan[1], 
        approximation.tspan[2], 
        reltol=results.solver.rtol, 
        abstol=results.solver.atol)[1]
end

function division_dist_hist(results::AnalyticalResults) 
    experiment = results.experiment
    approximation = results.approximation

    return hquadrature(
        length(results.birth_dist),
        (s,v) -> v[:] = 2 * 
            exp(-results.results[:growth_factor]* s) * 
            first_passage_time(
                results.results[:birth_dist], 
                s, 
                experiment.model_parameters, 
                results.cme_solution; 
                results=results), 
        approximation.tspan[1], 
        approximation.tspan[2], 
        reltol=results.solver.rtol, 
        abstol=results.solver.atol)[1]
end

function update_growth_factor!(results::AnalyticalResults)
    experiment = results.experiment
    approximation = results.approximation
    solver = results.solver

    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 
    marginalfpt(τ) = sum(
        first_passage_time(
            results.results[:birth_dist], 
            τ, 
            experiment.model_parameters, 
            results.cmesol; 
            results=results))

    # Root finding for the population growth rate λ.
    f(λ) = 1 - 2 *  hquadrature(
        s -> marginalfpt(s) * exp(-λ*s), 
        approximation.tspan[1], 
        approximation.tspan[2], 
        reltol=solver.rtol, 
        abstol=results.solver.atol)[1]

    results.results[:growth_factor] = find_zero(f, 0, solver.rootfinder)
end

function update_birth_dist!(results::AnalyticalResults)
    experiment = results.experiment
    approximation = results.approximation
    solver = results.solver
    model = results.model

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand(τ) = sum(
        partition.(
            model.partition_kernel,   
            collect.(axes(results.results[:birth_dist]))[1] .- 1, solver.truncation) 
        .* first_passage_time(
            results.results[:birth_dist], 
            τ, 
            experiment.model_parameters, 
            results.cmesol; 
            results=results) 
        .* 2 .*exp(-results.results[:growth_factor]*τ))

    results.results[:birth_dist] = hquadrature(
        length(results.results[:birth_dist]),
        (s,v) -> v[:] = boundary_cond_integrand(s), 
        approximation.tspan[1], 
        approximation.tspan[2], 
        reltol=results.solver.rtol,
        abstol=results.solver.atol)[1]

    results.results[:birth_dist] = results.results[:birth_dist] / sum(results.results[:birth_dist])
end

function log_convergece!(convergence::ConvergenceMonitor, results::AnalyticalResults)
    push!(convergence.birth_dists, results.birth_dist)
    push!(convergence.growth_factor, results.growth_factor)
end

function random_initial_values(approximation::AbstractAnalyticalApprox)
    truncation = prod(approximation.truncation)
    # Random normalised initial conditions.
    init = rand(1:truncation, truncation) 
    return  init ./ sum(init)
end

function solvecme(
        model::AbstractPopulationModel,
        experiment::AbstractExperimentSetup,
        approximation::AbstractAnalyticalApprox, 
        solver::AnalyticalSolverParameters)

    # Every interation refines birth_dist (Π(x|0)) and growth factor λ.
    results = AnalyticalResults()
    results.model = model
    results.solver = solver
    results.experiment = experiment
    results.approximation = approximation

    convergence = ConvergenceMonitor()
    results.results[:birth_dist] = random_initial_values(approximation)
    i::Int64 = 0

    progress = Progress(solver.maxiters;)
     
    while i < solver.maxiters
        # Solve the CME Π(x|τ).
        cme = cmemodel(results.results[:birth_dist], experiment.model_parameters, model, approximation)
        results.cmesol = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 

        update_growth_factor!(results)
        update_birth_dist!(results)
        #log_convergece!(convergence, results)

        i += 1
        ProgressMeter.next!(progress, showvalues = [("Current iteration", i), ("Growth factor λₙ", results.results[:growth_factor])])
    end

    cme = cmemodel(results.results[:birth_dist], experiment.model_parameters, model, approximation)
    cme_solution = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 
    results.cmesol = cme_solution
    results.convergence_monitor = convergence

    return results 
end
