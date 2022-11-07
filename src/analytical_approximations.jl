struct FiniteStateApprox <: AbstractAnalyticalApprox
    truncation::Vector{Int64}
    states
    function FiniteStateApprox(truncation)
        states = CartesianIndices(zeros(truncation...))
        states = map(x -> x.I .- tuple(I), states)

        return new(truncation, states)
    end
end

function cmemodel(
        xinit::Union{Matrix{Float64}, Vector{Float64}},
        parameters::Vector{Float64},
        tspan::Tuple{Float64, Float64},
        model::AbstractPopulationModel,
        approximation::FiniteStateApprox)

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approximation.truncation...), parameters, 0)
    #@time fA = convert(ODEFunction, fsp_problem)

    du = similar(xinit)
    Axu = similar(vec(xinit))
    divr = similar(xinit)

    function fu!(dx, x, p, τ)
#        fA.f(du, x, p, τ)
        mul!(Axu, A, vec(x))
        divr = divisionrate(approximation.states, p, τ, model.division_rate)
        dx .= reshape(Axu, size(xinit)) .- divr .* x
    end
    
    return ODEProblem(fu!, xinit, tspan, parameters)
end

function error(λ, results, approx::FiniteStateApprox)
    problem = results.problem
    model = results.problem.model

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    function ferr!(du, u, p, t)
        du[1] = bndA' * results.cmesol(t) * exp(-λ*t)
    end

    ferr_prob = ODEProblem(ferr!, [0.0,], problem.tspan,  []) 
    ferr_sol = solve(ferr_prob, results.solver.solver)

    return ferr_sol.u[end][1]
end

#function sanity_check(results, approx::FiniteStateApprox)
#    problem = results.problem
#    model = results.problem.model
#
#    fsp_problem = FSPSystem(model.molecular_model)
#    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
#
#    states = CartesianIndices(zeros(approx.truncation...))
#    states = map(x -> x.I .- tuple(I), states)
#
#    # Assuming no mass enters is added to the system the boundary states
#    # correspond to where the columns of A are negative.
#    bndA = abs.(vec(sum(A, dims=1)))
#
#    function f!(du, u, p, t)
#        du = A * u .- results.problem.model.division_rate.(states, fill(p, size(states)), t) .* u .- results.results[:growth_factor] .* u
#    end
#
#    u0 = results.results[:birth_dist] * 2 * results.results[:growth_factor]
#
#    f_prob = ODEProblem(f!, u0, problem.tspan,  problem.ps) 
#    f_sol = solve(f_prob, results.solver.solver)
#
#    function ferr!(du, u, p, t)
#        du[1] = bndA' * f_sol(t)
#    end
#
#    ferr_prob = ODEProblem(ferr!, [0.0,], problem.tspan, []) 
#    ferr_sol = solve(ferr_prob, results.solver.solver)
#
#    return ferr_sol.u[end] / results.results[:growth_factor]
#end

@inline function first_passage_time(
        x::Union{Matrix{Float64}, Matrix{Int64}, Vector{Float64}, Vector{Int64}}, 
        τ::Float64, 
        p::Vector{Float64}, 
        Π; 
        model,
        approx::FiniteStateApprox)

    # TODO: in general need to pass in also the reaction network to see which
    # indices correspond to which counts. 

    # First passage time for division.
    return max.(0.0, divisionrate(approx.states, p, τ, model.division_rate) .* Π(τ))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel)
    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    return (τ,p) -> sum(Ms .* first_passage_time(
            results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx) .* 2 .* exp(-results.results[:growth_factor]*τ))
end

function boundary_condition_ancest(problem, results, approx::FiniteStateApprox, model::CellPopulationModel)
    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    return (τ,p) -> sum(Ms .* first_passage_time(
            results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx))
end

function boundary_condition_alt(problem, results, approx::FiniteStateApprox, model::CellPopulationModel)
    problem = results.problem

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    return (τ,p) -> sum(Ms .* (first_passage_time(
            results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx) .+ bndA' * results.cmesol(τ)) .* 
                        2 .* exp(-results.results[:growth_factor]*τ))

end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::MotherCellModel)
    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    return (τ,p) -> sum(Ms .* first_passage_time(
            results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx) .* exp(-results.results[:growth_factor]*τ))
end

function boundary_condition_alt(problem, results, approx::FiniteStateApprox, model::MotherCellModel)
    problem = results.problem

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    return (τ,p) -> sum(Ms .* (first_passage_time(
            results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx) .+ bndA' * results.cmesol(τ)) .* 
                        exp(-results.results[:growth_factor]*τ))
end





