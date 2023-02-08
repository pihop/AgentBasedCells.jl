abstract type AbstractAnalyticalMethod end 

struct ToxicBoundaryDeath <: AbstractAnalyticalMethod end
struct ToxicBoundaryRe <: AbstractAnalyticalMethod end
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
    rootfinder::Union{Roots.AbstractSecant, Roots.AbstractBracketing}
    integrator
    bracket::Tuple{Float64, Float64}
    compute_joint::Bool

    function AnalyticalSolver(maxiters; 
            solver=Rodas5(autodiff=false), 
            rootfinder=Order16(), 
            integrator=QuadGKJL(),
            stol=1e-4, 
            rtol=1e-12, 
            atol=1e-12, 
            bracket=(0.0, 10.0),
            method=Reinsert(),
            compute_joint=false)

        new(method,maxiters, solver, stol, rtol, atol, rootfinder, integrator, bracket, compute_joint)
    end
end

mutable struct ConvergenceMonitor
    # TODO: Monitor convergence of the results. Here we want within the
    # iteration. Error control wrt truncation separately.
    birth_dists#::Array{Union{Matrix{Float64}, Vector{Float64}}}
    growth_factor::Array{Float64}

    function ConvergenceMonitor(birth_dist, lambda)
        new([birth_dist,], [lambda, ])
    end
end

mutable struct AnalyticalResults
    cmesol
    jointsol
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
    solver = results.solver
    integrand(u,p) = results.jointsol(u)
    prob = IntegralProblem(integrand, results.problem.tspan[1], results.problem.tspan[end], 0.0)
    telaps = @elapsed results.results[:marginal_size] = solve(prob, solver.integrator; reltol=rtol, abstol=atol).u
    println("Integration complete, took $telaps seconds.")
end

function mean_marginal_size(results::AnalyticalResults;)
    return sum(results.results[:marginal_size] .* collect(0:length(results.results[:marginal_size])-1))
end

function mode_size(results::AnalyticalResults;)
    return sum(results.results[:marginal_size] .* collect(0:length(results.results[:marginal_size])-1)) 
end

function division_time_dist(results::AnalyticalResults)
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    return τ -> sum(
        approx.fpt(results.results[:birth_dist], τ, results.cmesol, ratepre))
end

function fpt_dist(results::AnalyticalResults)
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    return τ -> approx.fpt(results.results[:birth_dist], τ, results.cmesol, ratepre)
end

function division_time_cdist!(results::AnalyticalResults; step)
    problem = results.problem 
    solver = results.solver
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    integrand(u,p) = sum(
        approx.fpt(results.results[:birth_dist], u, results.cmesol, ratepre))
    tspan = problem.tspan[1]:step:problem.tspan[2]
    prob(t1, t2) = IntegralProblem(integrand, t1, t2, 0.0)

    results.results[:division_time_cdist] = [
        solve(prob(t-step, t), solver.intergrator; reltol=solver.rtol, abstol=solver.atol).u for t in tspan[2:end]]
end

function joint_fpt_cdist!(results::AnalyticalResults; step)
    problem = results.problem 
    solver = results.solver
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    integrand(u,p) = approx.fpt(results.results[:birth_dist], u, results.cmesol, ratepre)
    tspan = problem.tspan[1]:step:problem.tspan[2]
    prob(t1, t2) = IntegralProblem(integrand, t1, t2, 0.0)

    results.results[:joint_fpt_cdist] = [
        solve(prob(t-step, t), solver.integrator; reltol=solver.rtol, abstol=solver.atol).u for t in tspan[2:end]]
end

function interdivision_time_dist(results::AnalyticalResults)
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    return τ -> 2*exp(-results.results[:growth_factor] * τ) * sum(
        approx.fpt(results.results[:birth_dist], u, results.cmesol, ratepre))
end

function division_dist!(results::AnalyticalResults) 
    problem = results.problem
    solver = results.solver
    approx = results.problem.approx
    ratepre = similar(results.results[:birth_dist])

    function integrand(u,p)
        approx.fpt(results.results[:birth_dist], u, results.cmesol, ratepre)
    end
    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    results.results[:division_dist] = solve(prob, solver.integrator, reltol=solver.rtol, abstol=solver.atol).u
end

function division_dist_hist!(results::AnalyticalResults) 
    problem = results.problem
    solver = results.solver
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])

    function integrand(u,p)
        2 * exp(-results.results[:growth_factor] * u) * 
            approx.fpt(results.results[:birth_dist], u, results.cmesol, ratepre)
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    results.results[:division_dist_ancest] = solve(prob, solver.integrator; callback=PositiveDomain(), reltol=solver.rtol, abstol=solver.atol).u
end

function birth_dist_hist!(results::AnalyticalResults)
    solver = results.solver
    problem = results.problem
    approx=results.problem.approx
    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    boundary_cond_integrand = boundary_condition_ancest(problem, results, problem.approx, problem.model)
    prob = IntegralProblem(boundary_cond_integrand, problem.tspan[1], problem.tspan[end])
    results.results[:birth_dist_ancest] = solve(prob, solver.integrator; callback=PositiveDomain(), reltol=solver.rtol, abstol=solver.atol).u
end


function update_growth_factor!(problem::AnalyticalProblem, model::MotherCellModel, results::AnalyticalResults, method, euler_lotka::Nothing)
    results.results[:growth_factor] = 0.0 
end

function update_growth_factor!(problem::AnalyticalProblem, 
        model::CellPopulationModel, 
        results::AnalyticalResults, 
        method::Union{Reinsert, ToxicBoundaryDeath},
        euler_lotka)
    solver = results.solver
    f(λ) = euler_lotka(λ, results.results[:birth_dist], results.cmesol)  
    try 
        results.results[:growth_factor] = find_zero(f, 0, solver.rootfinder; atol=solver.atol, rtol=solver.rtol, xatol = solver.atol, xrtol = solver.rtol)
    catch 
        @warn "Root finding failed. Terminating without solution"
        results.flag = :Failed
    end
end

function make_euler_lotka(problem::AnalyticalProblem, 
        model::MotherCellModel, 
        results::AnalyticalResults, 
        method::Union{Reinsert, ToxicBoundaryDeath})
    nothing
end

function make_euler_lotka(problem::AnalyticalProblem, 
        model::CellPopulationModel, 
        results::AnalyticalResults, 
        method::Union{Reinsert, ToxicBoundaryDeath})
    solver = results.solver
    ratepre = similar(results.results[:birth_dist])
    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 

    error_prob = error(results, problem.approx)
    correction(λ, bd, cmesol) = solve(remake(error_prob, p=[λ, bd, cmesol]), solver.solver; reltol=solver.rtol, abstol=solver.atol).u[end][1]

    function integrand(τ, p)
        return 2*exp(-p[1]*τ)*sum(problem.approx.fpt(p[2], τ, results.cmesol, ratepre))
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)

    (λ, bd, cmesol) -> 1 - solve(remake(prob, p=[λ,bd,cmesol]), solver.integrator; reltol=solver.rtol, abstol=solver.atol).u - 2*correction(λ, bd, cmesol)
end

function update_birth_dist!(problem::AnalyticalProblem, results::AnalyticalResults, method, boundary_cond)
    solver = results.solver
    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    ratepre = similar(results.results[:birth_dist])
    prob = IntegralProblem(boundary_cond, problem.tspan[1], problem.tspan[end], 
        [results.results[:growth_factor], results.results[:birth_dist], results.cmesol, ratepre])
    results.results[:birth_dist] = solve(prob, solver.integrator; reltol=solver.rtol, abstol=solver.atol).u
end

function log_convergece!(convergence::ConvergenceMonitor, results::AnalyticalResults)
    push!(convergence.birth_dists, results.results[:birth_dist])
    push!(convergence.growth_factor, results.results[:growth_factor])
end

function random_initial_values(approximation::AbstractAnalyticalApprox)
    truncation = approximation.truncation
    # Random normalised initial conditions.
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
    results.flag = :Success
    convergence = ConvergenceMonitor(results.results[:birth_dist], Inf)
    cme = cmemodel(results.results[:birth_dist], problem.ps, problem.tspan, problem.model, problem.approx)

    bnd_cond = boundary_condition(
            results.problem,
            results,
            results.problem.approx,
            results.problem.model,
            results.solver.method)

    euler_lotka = make_euler_lotka(problem,problem.model, results, solver.method)

    i::Int64 = 0
    progress = Progress(solver.maxiters;)

    changed = Inf
    changeλ = Inf
     
    while i < solver.maxiters && changed+changeλ > solver.stol
        # Solve the CME Π(x|τ).
        cme = remake(cme; u0=results.results[:birth_dist])
        results.cmesol = solve(cme, solver.solver; isoutofdomain=(y,p,t)->any(x->x<0,y), abstol=solver.atol, reltol=solver.rtol) 
        update_growth_factor!(problem,problem.model, results, solver.method, euler_lotka)

        if results.flag == :Success && results.cmesol.retcode == :Success
            update_birth_dist!(problem, results, solver.method, bnd_cond)

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
            results.flag = :Failed
            return results
        end
    end

    cme = remake(cme; u0=results.results[:birth_dist])
    if results.flag == :Success && solver.compute_joint
        results.cmesol = solve(cme, solver.solver; isoutofdomain=(y,p,t)->any(x->x<0,y), abstol=solver.atol) 

        initjoint = jointinit(results, results.problem.model)
        jointcme = jointmodel(results.results[:growth_factor], initjoint, problem.ps, problem.tspan, problem.model, problem.approx) 
        results.jointsol = solve(jointcme, solver.solver; isoutofdomain=(y,p,t)->any(x->x<0,y), abstol=solver.atol) 
    end

    results.convergence_monitor = convergence
    return results 
end
