abstract type AbstractAnalyticalMethod end 

struct ToxicBoundaryDeath <: AbstractAnalyticalMethod end
struct ToxicBoundaryRe <: AbstractAnalyticalMethod end
struct Reinsert <: AbstractAnalyticalMethod end
struct Divide <: AbstractAnalyticalMethod end

struct AnalyticalProblem{N}
    model::AbstractPopulationModel
    ps::NTuple{N, Float64}#Vector{Float64}
    tspan::Tuple{Float64, Float64}
    approx::AbstractAnalyticalApprox
end

function AnalyticalProblem(;model=model, ps=ps, tspan=tspan, approx=approx)
    return AnalyticalProblem(model, ps, tspan, approx)
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
#            integrator=QuadratureRule(FastGaussQuadrature.gausslegendre, n=100),
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
    monitor::Dict
    function ConvergenceMonitor()
        results = new()   
        results.monitor = Dict()
        return results 
    end
end

mutable struct AnalyticalResults
    cmesol
    jointsol
    results::Dict 
    problem::AnalyticalProblem
    euler_lotka
    error
    solver::AnalyticalSolver
    convergence_monitor::ConvergenceMonitor
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
    solver = results.solver
    
    integrand(u,p) = results.jointsol(u)
    prob = IntegralProblem(integrand, results.problem.tspan[1], results.problem.tspan[end], 0.0)

    results.results[:marginal_size] = solve(prob, solver.integrator; reltol=rtol, abstol=atol).u
    results.results[:marginal_size][results.results[:marginal_size] .< 0.0] .= 0.0
    results.results[:marginal_size] = results.results[:marginal_size] ./ sum(results.results[:marginal_size])
end

function marginal_age_distribution(results::AnalyticalResults)
    t -> sum(results.jointsol(t))
end

function marginal_gamma(results::AnalyticalResults)
    problem = results.problem
    marginal_age = marginal_age_distribution(results)
    drate = results.problem.model.division_rate
    truncation = results.problem.approx.truncation

    axes_ = collect.(Base.OneTo.(truncation))
    states = collect(Iterators.product(axes_...)) 

    function conditional(x, t)
        results.jointsol(t)[x...] / marginal_age(t)
    end
    
    t -> sum(divisionrate(map(x -> x .- tuple(I), states), problem.ps, t, drate) .* conditional.(states, Ref(t)))
end

function mean_marginal_size(results::AnalyticalResults;)
    return sum(results.results[:marginal_size] .* collect(0:length(results.results[:marginal_size])-1))
end

function mode_size(results::AnalyticalResults;)
    return sum(results.results[:marginal_size] .* collect(0:length(results.results[:marginal_size])-1)) 
end

function division_time_ancest(results::AnalyticalResults)
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])

    fpt = fpt_dist_ancest(results, results.problem.model)

    return τ -> sum(fpt(τ))
end

function division_hazard(results::AnalyticalResults)
    ddist = division_time_dist(results)
    solver = results.solver
    integrand(u,p) = ddist(u)

    prob(t) = IntegralProblem(integrand, 0, t, 0.0)
    cddist(t) = solve(prob(t), solver.integrator; reltol=solver.rtol, abstol=solver.atol).u 
    
    t -> ddist(t) / (1 - cddist(t))
end

function fpt_dist_ancest(results::AnalyticalResults)
    return fpt_dist_ancest(results, results.problem.model)
end

function fpt_dist_ancest(results::AnalyticalResults, model::CellPopulationModel)
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    return τ -> 2*exp(-results.results[:growth_rate]*τ) .* 
        approx.fpt(results.results[:birth_dist], τ, results.cmesol, ratepre)
end

function fpt_dist_ancest(results::AnalyticalResults, model::MotherCellModel)
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
    prob(t1, t2) = IntegralProblem(integrand, t1, t2)

    results.results[:division_time_cdist] = [
        solve(prob(t-step, t), solver.integrator; reltol=solver.rtol, abstol=solver.atol).u for t in tspan[2:end]]
end

function joint_fpt_ancest_cdist!(results::AnalyticalResults; step)
    problem = results.problem 
    solver = results.solver
    approx = results.problem.approx
    fpt = fpt_dist_ancest(results, problem.model)
    integrand(u,p) = fpt(u)
    tspan = problem.tspan[1]:step:problem.tspan[2]
    prob(t1, t2) = IntegralProblem(integrand, t1, t2)

    results.results[:joint_fpt_ancest_cdist] = [
        solve(prob(t-step, t), solver.integrator; reltol=solver.rtol, abstol=solver.atol).u for t in tspan[2:end]]
end

function interdivision_time_dist(results::AnalyticalResults)
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    return τ -> 2*exp(-results.results[:growth_rate] * τ) * sum(
        approx.fpt(results.results[:birth_dist], u, results.cmesol, ratepre))
end

function division_dist_leaf!(results::AnalyticalResults) 
    problem = results.problem
    solver = results.solver
    approx = results.problem.approx
    ratepre = similar(results.results[:birth_dist])

    function integrand(u,p)
        approx.fpt(results.results[:birth_dist], u, results.cmesol, ratepre)
    end
    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    results.results[:division_dist] = 
        solve(prob, solver.integrator, reltol=solver.rtol, abstol=solver.atol).u
end

function division_dist_ancest!(results::AnalyticalResults) 
    problem = results.problem
    solver = results.solver
    approx=results.problem.approx
    ratepre = similar(results.results[:birth_dist])
    fpt = fpt_dist_ancest(results, results.problem.model) 

    function integrand(u,p)
        fpt(u)
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    results.results[:division_dist_ancest] = solve(prob, solver.integrator; reltol=solver.rtol, abstol=solver.atol).u
end

#function birth_dist_ancest!(results::AnalyticalResults)
#    solver = results.solver
#    problem = results.problem
#    approx=results.problem.approx
#
#    ratepre = similar(results.results[:birth_dist])
#    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
#    boundary_cond_integrand = boundary_condition_ancest(problem, results, problem.approx, problem.model)
#    prob = IntegralProblem(boundary_cond_integrand, problem.tspan[1], problem.tspan[end],
#        [results.results[:growth_rate], results.results[:birth_dist], results.cmesol, ratepre])
#    results.results[:birth_dist_ancest] = solve(prob, solver.integrator; reltol=solver.rtol, abstol=solver.atol).u
#end


function update_growth_rate!(problem::AnalyticalProblem, model::MotherCellModel, results::AnalyticalResults, method, euler_lotka::Nothing)
    results.results[:growth_rate] = 0.0 
end

function update_growth_rate!(problem::AnalyticalProblem, 
        model::CellPopulationModel, 
        results::AnalyticalResults, 
        method::Union{Reinsert, Divide, ToxicBoundaryDeath},
        euler_lotka)
    solver = results.solver
    f(λ) = euler_lotka(λ, results.results[:birth_dist], results.cmesol)  
    try 
        results.results[:growth_rate] = find_zero(f, 0.0, solver.rootfinder; atol=solver.atol, rtol=solver.rtol)
    catch 
        @warn "Root finding failed. Terminating without solution"
        results.flag = :Failed
    end
end

function make_euler_lotka(problem::AnalyticalProblem, 
        model::MotherCellModel, 
        results::AnalyticalResults, 
        method::Union{Reinsert, Divide, ToxicBoundaryDeath})
    solver = results.solver
    ratepre = similar(results.results[:birth_dist])

    error_prob = error(results, problem.approx)
    correction(λ, bd, cmesol) = solve(remake(error_prob, p=(λ, bd, cmesol)), solver.solver; reltol=solver.rtol, abstol=solver.atol).u[end]

    results.euler_lotka = nothing

    error_(λ, bd, cmesol) = sum(cmesol(problem.tspan[end])) .+ correction(λ, bd, cmesol)
    results.error = error_
end

function make_euler_lotka(problem::AnalyticalProblem, 
        model::CellPopulationModel, 
        results::AnalyticalResults, 
        method::Union{Reinsert, Divide, ToxicBoundaryDeath})
    solver = results.solver
    ratepre = similar(results.results[:birth_dist])
    # Find marginal first passage time ν(t). This is normalised by
    # definition -- in the code up to numerical accuracy. 

    error_prob = error(results, problem.approx)
    correction(λ, bd, cmesol) = solve(remake(error_prob, p=(λ, bd, cmesol)), solver.solver; reltol=solver.rtol, abstol=solver.atol).u[end]

    function integrand(τ, p)
        return 2*exp(-p[1]*τ)*sum(problem.approx.fpt(p[2], τ, results.cmesol, ratepre))
    end

    prob = IntegralProblem(integrand, problem.tspan[1], problem.tspan[end], 0.0)
    results.euler_lotka = (λ, bd, cmesol) -> 
        1 - 
        exp(-λ*problem.tspan[end])*sum(cmesol(problem.tspan[end])) - 
        solve(remake(prob, p=(λ,bd,cmesol)), solver.integrator; reltol=solver.rtol, abstol=solver.atol).u - 
        correction(λ, bd, cmesol)[1]

    error_(λ, bd, cmesol) = [
        sum(cmesol(problem.tspan[end])) * exp(-λ*problem.tspan[end]), 
        sum(cmesol(problem.tspan[end]))] .+ correction(λ, bd, cmesol)
    results.error = error_
end

function update_birth_dist!(problem::AnalyticalProblem, results::AnalyticalResults, method, boundary_cond)
    solver = results.solver
    # Calculated new boundary condition given λ and CMEsol (Π(x|τ)).
    ratepre = similar(results.results[:birth_dist])
    prob = IntegralProblem(boundary_cond, problem.tspan[1], problem.tspan[end], 
        (results.results[:growth_rate], results.results[:birth_dist], results.cmesol, ratepre))
    results.results[:birth_dist] = solve(prob, solver.integrator; reltol=solver.rtol, abstol=solver.atol).u
end

function log_convergece!(convergence::ConvergenceMonitor, results::AnalyticalResults)
    push!(convergence.monitor[:birth_dist], results.results[:birth_dist])
    push!(convergence.monitor[:growth_rate], results.results[:growth_rate])
    push!(convergence.monitor[:division_dist], results.results[:division_dist])
end

function log_error!(problem::AnalyticalProblem, results::AnalyticalResults, error)
    err_ = error(results.results[:growth_rate], results.results[:birth_dist], results.cmesol)
    push!(results.results[:errors], err_)
end

function initial_dist(approximation::AbstractAnalyticalApprox)
    truncation = approximation.truncation
    zs = zeros(tuple(truncation...)) 
    zs[1] = 1.0
    return zs 
end

function solvecme(problem::AnalyticalProblem, solver::AnalyticalSolver)
    # Every interation refines birth_dist (Π(x|0)) and growth factor λ.
    results = AnalyticalResults()
    results.problem = problem
    results.solver = solver

    results.results[:birth_dist] = initial_dist(problem.approx)
    results.results[:birth_dist_iters] = [results.results[:birth_dist], ]
    results.results[:errors] = [] 
    results.results[:growth_rate] = Inf
    results.flag = :Success
    convergence = ConvergenceMonitor()
    convergence.monitor[:birth_dist] = [results.results[:birth_dist], ] 
    convergence.monitor[:division_dist] = [] 
    convergence.monitor[:growth_rate] = [Inf, ] 

    cme = cmemodel(results.results[:birth_dist], problem.ps, problem.tspan, problem.model, problem.approx)

    bnd_cond = boundary_condition(
            results.problem,
            results,
            results.problem.approx,
            results.problem.model,
            results.solver.method,
            results.solver.integrator)


    make_euler_lotka(problem, problem.model, results, solver.method)

    i::Int64 = 0
    progress = Progress(solver.maxiters;)

    changed = Inf
    changeλ = Inf
     
    while i < solver.maxiters && changed+changeλ > solver.stol
        # Solve the CME Π(x|τ).
        cme = remake(cme; u0=results.results[:birth_dist])
        
#        results.cmesol = solve(cme, solver.solver; isoutofdomain=(y,p,t)->any(x->x<0,y), abstol=solver.atol, reltol=solver.rtol)
        results.cmesol = solve(cme, solver.solver; abstol=solver.atol, reltol=solver.rtol)

        update_growth_rate!(problem, problem.model, results, solver.method, results.euler_lotka)

        if results.flag == :Success && SciMLBase.successful_retcode(results.cmesol)
            log_error!(problem, results, results.error)
            update_birth_dist!(problem, results, solver.method, bnd_cond)
            division_dist_leaf!(results)

            push!(results.results[:birth_dist_iters], results.results[:birth_dist])
            log_convergece!(convergence, results)

            i += 1
            changed = totalvariation(convergence.monitor[:birth_dist][end-1], convergence.monitor[:birth_dist][end])
            changeλ = abs(convergence.monitor[:growth_rate][end-1]-convergence.monitor[:growth_rate][end])

            ProgressMeter.next!(progress, 
                showvalues = [
                    ("Current iteration", i), 
                    ("Growth rate λ", results.results[:growth_rate]),
                    ("Running distance dist", changed),
                    ("Running distance λ", changeλ),
                    ("Total mass", sum(results.results[:birth_dist]))])
        else 
            results.flag = :Failed
            return results
        end
    end

    cme = remake(cme; u0=results.results[:birth_dist])
    if results.flag == :Success && solver.compute_joint
        compute_joint!(results)
    end
    results.results[:backwardjoint] = backwardjointmodel(
        results.results[:birth_dist], 
        results.results[:growth_rate], 
        problem.ps, 
        problem.tspan, 
        problem.model, 
        problem.approx)

    results.convergence_monitor = convergence
    return results 
end

function compute_joint!(results::AnalyticalResults)
    solver = results.solver
    problem = results.problem
    initjoint = jointinit(results, results.problem.model)
    jointcme = jointmodel(initjoint, results.results[:growth_rate], problem.ps, problem.tspan, problem.model, problem.approx) 
#    results.jointsol = solve(jointcme, solver.solver; isoutofdomain=(y,p,t)->any(x->x<0,y), abstol=solver.atol) 
    results.jointsol = solve(jointcme, solver.solver; abstol=solver.atol) 
end
