struct FiniteStateApprox <: AbstractAnalyticalApprox
    truncation::Vector{Int64}
end

function cmemodel(
        xinit::Vector{Float64},
        parameters::Vector{Float64},
        tspan::Tuple{Float64, Float64},
        model::CellPopulationModel,
        approximation::FiniteStateApprox)

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approximation.truncation...), parameters, 0)

    states = CartesianIndices(zeros(approximation.truncation...))
    states = map(x -> x.I .- tuple(I), states)

    function fu!(dx, x, p, τ)
        dx[:] = A * x - model.division_rate.(states, fill(p, size(states)), τ) .* x
    end

    return ODEProblem(fu!, xinit, tspan, parameters)
end

function error(results, approx::FiniteStateApprox)
    problem = results.problem
    model = results.problem.model

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    function ferr!(du, u, p, t)
        du[1] = bndA' * results.cmesol(t) 
    end

    ferr_prob = ODEProblem(ferr!, [0.0,], problem.tspan,  []) 
    ferr_sol = solve(ferr_prob, results.solver.solver)

    return ferr_sol.u[end][1]
end

function sanity_check(results, approx::FiniteStateApprox)
    problem = results.problem
    model = results.problem.model

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)

    states = CartesianIndices(zeros(approx.truncation...))
    states = map(x -> x.I .- tuple(I), states)

    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    function f!(du, u, p, t)
        du = A * u .- results.problem.model.division_rate.(states, fill(p, size(states)), t) .* u .- results.results[:growth_factor] .* u
    end

    u0 = results.results[:birth_dist] * 2 * results.results[:growth_factor]

    f_prob = ODEProblem(f!, u0, problem.tspan,  problem.ps) 
    f_sol = solve(f_prob, results.solver.solver)

    function ferr!(du, u, p, t)
        du[1] = bndA' * f_sol(t)
    end

    ferr_prob = ODEProblem(ferr!, [0.0,], problem.tspan, []) 
    ferr_sol = solve(ferr_prob, results.solver.solver)

    return ferr_sol.u[end] / results.results[:growth_factor]
end

function first_passage_time(
        x::Union{Vector{Float64}, Vector{Int64}}, 
        τ::Real, 
        p::Vector{Float64}, 
        Π; 
        model,
        approx::FiniteStateApprox)

    # TODO: in general need to pass in also the reaction network to see which
    # indices correspond to which counts. 

    states = CartesianIndices(zeros(approx.truncation...))
    states = map(x -> x.I .- tuple(I), states)

    # First passage time for division.
    return model.division_rate.(states, fill(p, size(states)), τ) .* Π(τ) 
end

function boundary_condition(problem, results, approx::FiniteStateApprox)
    return τ -> sum(
        partition.(
            problem.model.partition_kernel,   
            collect.(axes(results.results[:birth_dist]))[1] .- 1, approx.truncation) 
        .* first_passage_time(
            results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx) 
        .* 2 .*exp(-results.results[:growth_factor]*τ))
end

