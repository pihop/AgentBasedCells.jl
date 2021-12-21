struct AnalyticalModel
    xinit::Vector{Float64}
    parameters::Vector{Float64}
    tspan::Tuple{Float64, Float64}
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

    function AnalyticalSolverParameters(truncation, maxiters;solver=Vern7(), rootfinder=Order1(), rtol=1e-8, atol=1e-8)
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
    birth_dist::Vector{Float64}
    division_dist::Vector{Float64}
    growth_factor::Float64 
    cme_solution
    model::AnalyticalModel
    solver::AnalyticalSolverParameters
    convergence_monitor::ConvergenceMonitor

    function AnalyticalResults()
        new()
    end
end

function first_passage_time(x::Union{Vector{Float64}, Vector{Int64}}, 
    τ::Float64, p::Vector{Float64}, Π; results)
    # TODO: in general need to pass in also the reaction network to see which
    # indices correspond to which counts. 
    model = results.model

    states = CartesianIndices(zeros(results.solver.truncation...))
    states = map(x -> x.I .- tuple(I), states)
    
    # First passage time for division.
    return model.division_rate.(states, τ, fill(p, size(states))) .* Π(τ) 
end

function division_time_dist(results::AnalyticalResults)
    model = results.model
    return τ -> sum(
        first_passage_time(results.birth_dist, τ, model.parameters, results.cme_solution; results=results))
end

function division_dist(results::AnalyticalResults) 
    model = results.model
    return quadgk(
        s -> first_passage_time(results.birth_dist, s, model.parameters, results.cme_solution; results=results), 
        model.tspan[1], model.tspan[2], rtol=solver.rtol)[1]
end

function division_dist_hist(results::AnalyticalResults) 
    model = results.model
    return quadgk(
        s -> 2 * 
            exp(-results.growth_factor * s) * 
            first_passage_time(results.birth_dist, s, model.parameters, results.cme_solution; results=results), 
        model.tspan[1], model.tspan[2], rtol=solver.rtol)[1]
end

function update_growth_factor!(model::AnalyticalModel, results::AnalyticalResults, 
    solver::AnalyticalSolverParameters, cme_solution::ODESolution)
    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 
    marginalfpt(τ) = sum(
        first_passage_time(results.birth_dist, τ, model.parameters, cme_solution; results=results))

    # Root finding for the population growth rate λ.
    f(λ) = 1 - 2 *  quadgk(s -> marginalfpt(s) * exp(-λ*s), model.tspan[1], model.tspan[2], rtol=solver.rtol)[1]
    results.growth_factor = find_zero(f, 0, solver.rootfinder)
end

function update_birth_dist!(model::AnalyticalModel, results::AnalyticalResults, 
    solver::AnalyticalSolverParameters, cme_solution::ODESolution)

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand(τ) = sum(
        model.partition_kernel.(collect.(axes(results.birth_dist))[1] .- 1, solver.truncation) .* 
        first_passage_time(results.birth_dist, τ, model.parameters, cme_solution; results=results) .* 2 .* exp(-results.growth_factor*τ))
    results.birth_dist = quadgk(s -> boundary_cond_integrand(s), model.tspan[1], model.tspan[2], rtol=1e-2)[1]
    results.birth_dist = results.birth_dist / sum(results.birth_dist)
end

function log_convergece!(convergence::ConvergenceMonitor, results::AnalyticalResults)
    push!(convergence.birth_dists, results.birth_dist)
    push!(convergence.growth_factor, results.growth_factor)
end

function solvecme(model::AnalyticalModel,
    solver::AnalyticalSolverParameters)

    # Every interation refines birth_dist (Π(x|0)) and growth factor λ.
    results = AnalyticalResults()
    results.model = model
    results.solver = solver

    convergence = ConvergenceMonitor()
    results.birth_dist = model.xinit
    i::Int64 = 0

    progress = Progress(solver.maxiters;)
     
    while i < solver.maxiters
        # Solve the CME Π(x|τ).
        cme = model.cme_model(results.birth_dist, model.tspan, model.parameters, solver.truncation)
        cme_solution = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 

        update_growth_factor!(model, results, solver, cme_solution)
        update_birth_dist!(model, results, solver, cme_solution)
        log_convergece!(convergence, results)

        i += 1
        ProgressMeter.next!(progress, showvalues = [("Current iteration", i), ("Growth factor λₙ", results.growth_factor)])
    end

    cme = model.cme_model(results.birth_dist, model.tspan, model.parameters, solver.truncation)
    cme_solution = solve(cme, solver.solver; cb=PositiveDomain(), atol=solver.atol) 
    results.cme_solution = cme_solution
    results.convergence_monitor = convergence

    return results 
end


