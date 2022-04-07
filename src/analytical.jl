struct AnalyticalProblem
    model::AbstractPopulationModel
    init::Vector{Float64}
    ps::Vector{Float64}
    tspan::Tuple{Float64, Float64}
    approx::AbstractAnalyticalApprox
end

struct AnalyticalSolver
    maxiters::Int64
    solver::Any
    rtol::Float64
    atol::Float64
    rootfinder::Roots.AbstractSecant

    function AnalyticalSolver(maxiters; solver=Vern7(), rootfinder=Order1(), rtol=1e-8, atol=1e-8)
        new(maxiters, solver, rtol, atol, rootfinder)
    end
end

mutable struct ConvergenceMonitor
    # TODO: Monitor convergence of the results. Here we want within the
    # iteration. Error control wrt truncation separately.
    birth_dists::Array{Vector{Float64}}
    growth_factor::Array{Float64}

    function ConvergenceMonitor(birth_dist)
        new([birth_dist,], [])
    end
end

mutable struct AnalyticalResults
    cmesol
    results::Dict 
    problem::AnalyticalProblem
    solver::AnalyticalSolver
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
        hquadrature(s -> division_time_dist(results)(s), τ, results.problem.tspan[end]; abstol=atol)[1]
    
    telaps = @elapsed marginalΠ = hquadrature(
        length(results.results[:birth_dist]), 
        (s,v) -> v[:] = results.cmesol(s) * Π(s), 
        results.problem.tspan[1], 
        results.problem.tspan[end]; abstol=atol)[1]

    results.results[:marginal_size] = marginalΠ ./ sum(marginalΠ)
    println("Integration complete, took $telaps seconds.")
end

function mean_marginal_size(results::AnalyticalResults;)
    return sum(results.results[:marginal_size] .* collect(0:length(results.results[:marginal_size])-1)) 
end

function division_time_dist(results::AnalyticalResults)
    return τ -> sum(
        first_passage_time(
            results.results[:birth_dist], 
            τ, 
            results.problem.ps, 
            results.cmesol; 
            model=results.problem.model,
            approx=results.problem.approx))
end

function division_dist(results::AnalyticalResults) 
    return hquadrature(
        length(results.results[:birth_dist]),
        (s,v) -> v[:] = first_passage_time(
        results.results[:birth_dist], 
            s, 
            model.problem.model_parameters, 
            results.cmesol; 
            results=results), 
        approximation.tspan[1], 
        approximation.tspan[2], 
        reltol=results.solver.rtol, 
        abstol=results.solver.atol)[1]
end

function division_dist_hist(results::AnalyticalResults) 
    return hquadrature(
        length(results.results[:birth_dist]),
        (s,v) -> v[:] = 2 * 
            exp(-results.results[:growth_factor]* s) * 
            first_passage_time(
                results.results[:birth_dist], 
                s, 
                results.problem.ps, 
                results.cmesol; 
                model=results.problem.model,
                approx=results.problem.approx
               ), 
        results.problem.tspan[1], 
        results.problem.tspan[end], 
        reltol=results.solver.rtol, 
        abstol=results.solver.atol)[1]
end

function update_growth_factor!(problem::AnalyticalProblem, results::AnalyticalResults)
    solver = results.solver

    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 
    marginalfpt(τ) = sum(
        first_passage_time(
            results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx))

    # Root finding for the population growth rate λ.
    f(λ) = 1 - 2 *  hquadrature(
        s -> marginalfpt(s) * exp(-λ*s), 
        problem.tspan[1], 
        problem.tspan[end], 
        reltol=solver.rtol, 
        abstol=results.solver.atol)[1]

    results.results[:growth_factor] = find_zero(f, 0, solver.rootfinder)
end

function update_birth_dist!(problem::AnalyticalProblem, results::AnalyticalResults)
    solver = results.solver

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand = boundary_condition(problem, results, problem.approx)

    results.results[:birth_dist] = hquadrature(
        length(results.results[:birth_dist]),
        (s,v) -> v[:] = boundary_cond_integrand(s), 
        problem.tspan[1], 
        problem.tspan[end], 
        reltol=results.solver.rtol,
        abstol=results.solver.atol)[1]

    results.results[:birth_dist] = results.results[:birth_dist] / sum(results.results[:birth_dist])
end

function log_convergece!(convergence::ConvergenceMonitor, results::AnalyticalResults)
    push!(convergence.birth_dists, results.results[:birth_dist])
    push!(convergence.growth_factor, results.results[:growth_factor])
end

function random_initial_values(approximation::AbstractAnalyticalApprox)
    truncation = prod(approximation.truncation)
    # Random normalised initial conditions.
    init = rand(1:truncation, truncation) 
    return  init ./ sum(init)
end

function solvecme(problem::AnalyticalProblem, solver::AnalyticalSolver)

    # Every interation refines birth_dist (Π(x|0)) and growth factor λ.
    results = AnalyticalResults()
    results.problem = problem
    results.solver = solver

    results.results[:birth_dist] = random_initial_values(problem.approx)
    convergence = ConvergenceMonitor(results.results[:birth_dist])
    i::Int64 = 0

    progress = Progress(solver.maxiters;)
     
    while i < solver.maxiters
        # Solve the CME Π(x|τ).
        cme = cmemodel(results.results[:birth_dist], problem.ps, problem.tspan, problem.model, problem.approx)
        results.cmesol = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 

        update_growth_factor!(problem, results)
        update_birth_dist!(problem, results)
        log_convergece!(convergence, results)

        i += 1
        ProgressMeter.next!(progress, 
            showvalues = [
                ("Current iteration", i), 
                ("Growth factor λₙ", results.results[:growth_factor]),
                ("Running distance", kl_divergence(convergence.birth_dists[end-1], convergence.birth_dists[end] .+ 1e-4))])
    end

    cme = cmemodel(results.results[:birth_dist], problem.ps, problem.tspan, problem.model, problem.approx)
    cme_solution = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 
    results.cmesol = cme_solution
    results.convergence_monitor = convergence

    return results 
end
