abstract type AbstractAnalyticalMethod end 

struct ToxicBoundaryUpper <: AbstractAnalyticalMethod end
struct ToxicBoundaryLower <: AbstractAnalyticalMethod end
struct Reinsert <: AbstractAnalyticalMethod end


struct AnalyticalProblem
    model::AbstractPopulationModel
    ps::Vector{Float64}
    tspan::Tuple{Float64, Float64}
    approx::AbstractAnalyticalApprox
end

struct AnalyticalSolver
    method::AbstractAnalyticalMethod
    maxiters::Int64
    solver::Any
    stol::Float64
    rtol::Float64
    atol::Float64
    rootfinder::Roots.AbstractSecant
    bracket::Tuple{Float64, Float64}

    function AnalyticalSolver(maxiters; 
            solver=AutoTsit5(Rosenbrock23(autodiff=false)), 
            rootfinder=Order1(), 
            stol=1e-3, 
            rtol=1e-12, 
            atol=1e-12, 
            bracket=(0.0, 1000.0),
            method=ToxicBoundaryLower())

        new(method,maxiters, solver, stol, rtol, atol, rootfinder, bracket)
    end
end

mutable struct ConvergenceMonitor
    # TODO: Monitor convergence of the results. Here we want within the
    # iteration. Error control wrt truncation separately.
    birth_dists::Array{Union{Matrix{Float64}, Vector{Float64}}}
    growth_factor::Array{Float64}

    function ConvergenceMonitor(birth_dist, lambda)
        new([birth_dist,], [lambda, ])
    end
end

mutable struct AnalyticalResults
    cmesol
    results::Dict 
    problem::AnalyticalProblem
    solver::AnalyticalSolver
    convergence_monitor::ConvergenceMonitor
    error::Float64
    flag

    function AnalyticalResults()
        results = new()
        results.results = Dict()
        return results
    end
end

function Base.show(io::IO, ::AnalyticalResults)
    print(io, "Analytical computation results for cell population model.")
end

function marginal_size_distribution!(results::AnalyticalResults; rtol=1e-6, atol=1e-6)
    println("Calculating marginal size ...")
    λ = results.results[:growth_factor]
    Π(τ) = 2*λ*exp(-λ*τ)*
        hquadrature(s -> division_time_dist(results)(s), τ, results.problem.tspan[end]; abstol=atol)[1]
    
    telaps = @elapsed marginalΠ = hquadrature(
        length(results.results[:birth_dist]), 
        (s,v) -> v[:] = results.cmesol(s) * Π(s), 
        results.problem.tspan[1], 
        results.problem.tspan[end]; abstol=atol)[1]

    marginalΠ = max.(Ref(0.0), marginalΠ)

    results.results[:marginal_size] = marginalΠ ./ sum(marginalΠ)
    println("Integration complete, took $telaps seconds.")
end

function mean_marginal_size(results::AnalyticalResults;)
    return max.(Ref(0.0), sum(results.results[:marginal_size] .* collect(0:length(results.results[:marginal_size])-1)))
end

function mode_size(results::AnalyticalResults;)
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

function division_time_cdist!(results::AnalyticalResults; step)
    problem = results.problem 
    solver = results.solver

    integrand(u,p) = sum(
        first_passage_time(
            results.results[:birth_dist], 
            u, 
            results.problem.ps, 
            results.cmesol; 
            model=results.problem.model,
            approx=results.problem.approx))

    tspan = problem.tspan[1]:step:problem.tspan[2]
    prob(t1, t2) = IntegralProblem(integrand, t1, t2, 0.0)

    results.results[:division_time_cdist] = [
        solve(prob(t-step, t), QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u for t in tspan[2:end]]
end

function joint_fpt_cdist!(results::AnalyticalResults; step)
    problem = results.problem 
    solver = results.solver

    integrand(u,p) = first_passage_time(
            results.results[:birth_dist], 
            u, 
            results.problem.ps, 
            results.cmesol; 
            model=results.problem.model,
            approx=results.problem.approx)

    tspan = problem.tspan[1]:step:problem.tspan[2]
    prob(t1, t2) = IntegralProblem(integrand, t1, t2, 0.0)

    results.results[:joint_fpt_cdist] = [
        max.(0.0, solve(prob(t-step, t), QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u) for t in tspan[2:end]]
end

function interdivision_time_dist(results::AnalyticalResults)
    return τ -> 2*exp(-results.results[:growth_factor] * τ) * sum(
        first_passage_time(
            results.results[:birth_dist], 
            τ, 
            results.problem.ps, 
            results.cmesol; 
            model=results.problem.model,
            approx=results.problem.approx))
end

function division_dist!(results::AnalyticalResults) 
    problem = results.problem
    solver = results.solver

    function integrand(u,p)
        first_passage_time(
            results.results[:birth_dist], 
                u, 
                results.problem.ps, 
                results.cmesol; 
                model=results.problem.model,
                approx=results.problem.approx)
    end
    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    results.results[:division_dist] = solve(prob, QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
end

function division_dist_hist!(results::AnalyticalResults) 
    problem = results.problem
    solver = results.solver

    function integrand(u,p)
        2 * exp(-results.results[:growth_factor] * u) * 
            first_passage_time(
                results.results[:birth_dist], 
                u, 
                results.problem.ps, 
                results.cmesol; 
                model=results.problem.model,
                approx=results.problem.approx
               )
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    results.results[:division_dist_ancest] = solve(prob, QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
end

function birth_dist_hist!(results::AnalyticalResults)
    solver = results.solver
    problem = results.problem
    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand = boundary_condition_ancest(problem, results, problem.approx, problem.model)
    prob = IntegralProblem(boundary_cond_integrand, problem.tspan[1], problem.tspan[end])
    results.results[:birth_dist_ancest] = solve(prob, QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
end


function update_growth_factor!(problem::AnalyticalProblem, model::MotherCellModel, results::AnalyticalResults, method)
    results.results[:growth_factor] = 0.0 
end

function update_growth_factor!(problem::AnalyticalProblem, model::CellPopulationModel, results::AnalyticalResults, method::Reinsert)
     solver = results.solver

    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 
    marginalfpt(u) = sum(
        first_passage_time(
            results.results[:birth_dist], 
            u, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx))

    function integrand(u,p)
        return exp(-p*u)*marginalfpt(u)
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    f(λ) = 1 - 2*solve(remake(prob, p=λ), QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
    try 
        results.results[:growth_factor] = find_zero(f, 0, solver.rootfinder; atol=solver.atol, rtol=solver.rtol)
    catch
        @warn "Root finding failed. Terminating without solution"
        results.flag = :failed
    end
   
end

function update_growth_factor!(problem::AnalyticalProblem,  model::CellPopulationModel, results::AnalyticalResults, method::ToxicBoundaryLower;)
    solver = results.solver

    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 
    marginalfpt(u) = sum(
        first_passage_time(
            results.results[:birth_dist], 
            u, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx))

    function integrand(u,p)
        return exp(-p*u)*marginalfpt(u)
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    f(λ) = 1 - 2*solve(remake(prob, p=λ), QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
    try 
        results.results[:growth_factor] = find_zero(f, 0, solver.rootfinder; atol=solver.atol, rtol=solver.rtol)
    catch
        @warn "Root finding failed. Terminating without solution"
        results.flag = :failed
    end
end

function update_growth_factor!(problem::AnalyticalProblem, model::CellPopulationModel, results::AnalyticalResults, method::ToxicBoundaryUpper;)
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


    correction(p) = error(p, results, problem.approx)

    function integrand(u,p)
        return marginalfpt(u) + correction(p)
    end

    function integrand_err(u,p)
        return exp(-p*u)*correction(p)
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    prob_err = IntegralProblem(integrand_err, problem.tspan[1], problem.tspan[end], 0.0)
    f(λ) = sum(results.results[:birth_dist]) - 
        2*solve(remake(prob, p=λ), QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
    try 
        results.results[:growth_factor] = find_zero(f, 1, solver.rootfinder; atol=solver.atol, rtol=solver.rtol)
        push!(results.results[:errors], 
            solve(remake(prob_err, p=results.results[:growth_factor]), QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u)
    catch
        @warn "Root finding failed. Terminating without solution"
        results.flag = :failed
    end
end

function update_birth_dist!(problem::AnalyticalProblem, results::AnalyticalResults, method::ToxicBoundaryLower)
    solver = results.solver

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand = boundary_condition(problem, results, problem.approx, problem.model)
    prob = IntegralProblem(boundary_cond_integrand, problem.tspan[1], problem.tspan[end])
    results.results[:birth_dist] = solve(prob, QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
end

function update_birth_dist!(problem::AnalyticalProblem, results::AnalyticalResults, method::ToxicBoundaryUpper)
    solver = results.solver

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand = boundary_condition(problem, results, problem.approx, problem.model)
    prob = IntegralProblem(boundary_cond_integrand, problem.tspan[1], problem.tspan[end])
    results.results[:birth_dist] = solve(prob, QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
#    results.results[:birth_dist_non_norm] = results.results[:birth_dist]
    results.results[:birth_dist] = results.results[:birth_dist] ./ sum(results.results[:birth_dist])
end

function update_birth_dist!(problem::AnalyticalProblem, results::AnalyticalResults, method::Reinsert)
    solver = results.solver

    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand = boundary_condition_alt(problem, results, problem.approx, problem.model)
    prob = IntegralProblem(boundary_cond_integrand, problem.tspan[1], problem.tspan[end])
    results.results[:birth_dist] = solve(prob, QuadGKJL(); reltol=solver.rtol, abstol=solver.atol).u
#    results.results[:birth_dist_non_norm] = results.results[:birth_dist]
#    results.results[:birth_dist] = results.results[:birth_dist] ./ sum(results.results[:birth_dist])
end

function log_convergece!(convergence::ConvergenceMonitor, results::AnalyticalResults)
    push!(convergence.birth_dists, results.results[:birth_dist])
    push!(convergence.growth_factor, results.results[:growth_factor])
end

function random_initial_values(approximation::AbstractAnalyticalApprox)
    truncation = approximation.truncation
    #truncation = prod(approximation.truncation)
    # Random normalised initial conditions.
#    init = rand(1:truncation, truncation) 
    init = rand(truncation...)
    return  init ./ sum(init)
end

function solvecme(problem::AnalyticalProblem, solver::AnalyticalSolver)
    # Every interation refines birth_dist (Π(x|0)) and growth factor λ.
    results = AnalyticalResults()
    results.problem = problem
    results.solver = solver

    results.results[:birth_dist] = random_initial_values(problem.approx)
    results.results[:birth_dist_iters] = [results.results[:birth_dist], ]
    results.results[:errors] = [] 
    results.flag = :success
    convergence = ConvergenceMonitor(results.results[:birth_dist], Inf)
    cme = cmemodel(results.results[:birth_dist], problem.ps, problem.tspan, problem.model, problem.approx)

    i::Int64 = 0
    progress = Progress(solver.maxiters;)

    changed = Inf
    changeλ = Inf
     
    while i < solver.maxiters && changed+changeλ > solver.stol
        # Solve the CME Π(x|τ).
        #cme = remake(cme; u0=vcat(results.results[:birth_dist], 0., 1.0))
        cme = remake(cme; u0=results.results[:birth_dist])
        try
            results.cmesol = solve(cme, solver.solver; callback=PositiveDomain(), abstol=solver.atol) 
        catch
            @warn "CME solution failed. Terminating without solution"
            results.flag = :failed
            break
        end

        update_growth_factor!(problem,problem.model,results, solver.method)

        if results.flag != :failed
            update_birth_dist!(problem, results, solver.method)
            push!(results.results[:birth_dist_iters], results.results[:birth_dist])
            log_convergece!(convergence, results)

            i += 1
            changed = totalvariation(convergence.birth_dists[end-1], convergence.birth_dists[end])
            changeλ = abs(convergence.growth_factor[end-1]-convergence.growth_factor[end])

            ProgressMeter.next!(progress, 
                showvalues = [
                    ("Current iteration", i), 
                    ("Growth factor λ", results.results[:growth_factor]),
                    ("Running distance dist", changed),
                    ("Running distance λ", changeλ)])
        else 
            return results
        end
    end

    cme = remake(cme; u0=results.results[:birth_dist])
    if results.flag != :failed
        results.cmesol = solve(cme, solver.solver; callback=PositiveDomain(), abstol=solver.atol) 
    end
    results.convergence_monitor = convergence
    return results 
end
