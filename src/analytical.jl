struct AnalyticalModel
    xinit::Vector{Float64}
    parameters::Vector{Float64}
    tspan::Tuple{Float64, Float64}
    cme_model::Function
    partition_kernel::Function
    first_passage_time::Function
end

struct AnalyticalSolverParameters
    truncation::Int64
    maxiters::Int64
    solver::OrdinaryDiffEqAlgorithm
    rtol::Float64
    atol::Float64
    rootfinder::Roots.AbstractSecant

    function AnalyticalSolverParameters(truncation, maxiters;solver=Vern7(), rootfinder=Order1(), rtol=1e-8, atol=1e-8)
        new(truncation, maxiters, solver, rtol, atol, rootfinder)
    end
end

mutable struct ConvergenceMonitor
    # TODO: Monitor convergence of the results. We want both within the
    # iteration and for different tructations.
end

mutable struct AnalyticalResults
    # TODO: Structure holding the most necessary. Implement derived calculations
    # based on thist struct.
    #parameters::Vector{Float64}
    #cme_model::Function
    #division_time_dist::Function
    birth_dist::Vector{Float64}
    division_dist::Vector{Float64}
    growth_factor::Float64 
    model::AnalyticalModel
    solver::AnalyticalSolverParameters
    convergence_monitor::ConvergenceMonitor

    function AnalyticalResults()
        new()
    end
end

function division_time_dist(results::AnalyticalResults)
    model = results.model
    solver = results.solver

    cme = model.cme_model(results.birth_dist, model.tspan, model.parameters, solver.truncation)
    cme_solution = solve(cme, solver.solver; cb=PositiveDomain())# isoutofdomain=(u,p,t) -> any(x -> x .< 0, u), atol=1e-8) 
    return τ -> sum(model.first_passage_time(results.birth_dist, τ, model.parameters, cme_solution))
end

function division_dist(results::AnalyticalResults) 
    model = results.model
    solver = results.solver

    cme = model.cme_model(results.birth_dist, model.tspan, model.parameters, solver.truncation)
    cme_solution = solve(cme, solver.solver; cb=PositiveDomain())# isoutofdomain=(u,p,t) -> any(x -> x .< 0, u), atol=1e-8) 
#    division_time_dist(τ) = sum(first_passage_time(results.birth_dist, τ, parameters, cme_star_solution))
#    div_time_dist(τ) = division_time_dist(results)
    return quadgk(
        s -> 2 * exp(-results.growth_factor*s) * model.first_passage_time(results.birth_dist, s, model.parameters, cme_solution), 
        model.tspan[1], model.tspan[2], rtol=solver.rtol)[1]
end

function update_growth_factor!(model::AnalyticalModel, results::AnalyticalResults, 
    solver::AnalyticalSolverParameters, cme_solution::ODESolution)
    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 
    marginalfpt(τ) = sum(model.first_passage_time(results.birth_dist, τ, model.parameters, cme_solution))

    # Root finding for the population growth rate λ.
    f(λ) = 1 - 2 *  quadgk(s -> marginalfpt(s) * exp(-λ*s), model.tspan[1], model.tspan[2], rtol=solver.rtol)[1]
    results.growth_factor = find_zero(f, 0, solver.rootfinder)
end

function update_birth_dist!(model::AnalyticalModel, results::AnalyticalResults, 
    solver::AnalyticalSolverParameters, cme_solution::ODESolution)

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand(τ) = sum(
        model.partition_kernel.(collect.(axes(results.birth_dist))[1] .- 1, solver.truncation) .* 
        model.first_passage_time(results.birth_dist, τ, model.parameters, cme_solution) .* 2 .* exp(-results.growth_factor*τ))
    
    results.birth_dist = quadgk(s -> boundary_cond_integrand(s), model.tspan[1], model.tspan[2], rtol=solver.rtol)[1]
    results.birth_dist = results.birth_dist / sum(results.birth_dist)
end

function solvecme(model::AnalyticalModel,
    solver::AnalyticalSolverParameters)

    # Every interation refines birth_dist (Π(x|0)) and growth factor λ.
    results = AnalyticalResults()
    results.birth_dist = model.xinit
    i::Int64 = 0

    progress = Progress(solver.maxiters;)
     
    while i < solver.maxiters
        # Solve the CME Π(x|τ).
        cme = model.cme_model(results.birth_dist, model.tspan, model.parameters, solver.truncation)
        cme_solution = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 

        update_growth_factor!(model, results, solver, cme_solution)
        update_birth_dist!(model, results, solver, cme_solution)

        i += 1
        ProgressMeter.next!(progress, showvalues = [("Current iteration", i), ("Growth factor λₙ", results.growth_factor)])
    end

    results.model = model
    results.solver = solver

    return results 
end


