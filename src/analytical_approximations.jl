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
        xinit,
        parameters::Vector{Float64},
        tspan::Tuple{Float64, Float64},
        model::AbstractPopulationModel,
        approximation::FiniteStateApprox)

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approximation.truncation...), parameters, 0)

    Axu = similar(vec(xinit))
    divr = similar(xinit)

    function fu!(dx, x, p, τ)
        mul!(Axu, A, vec(x))
        divr = divisionrate(approximation.states, p, τ, model.division_rate)
        dx .= reshape(Axu, size(xinit)) .- divr .* x
    end
    
    return ODEProblem(fu!, xinit, tspan, parameters)
end

function error(λ, cmesol, results, approx::FiniteStateApprox)
    problem = results.problem
    model = results.problem.model

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    function ferr!(du, u, p, t)
        du[1] = bndA' * vec(cmesol(t)) * exp(-λ*t)
    end

    ferr_prob = ODEProblem(ferr!, [0.0,], problem.tspan,  []) 
    ferr_sol = solve(ferr_prob, results.solver.solver; abstol=results.solver.atol, reltol=results.solver.rtol)

    return ferr_sol.u[end][1]
end

@inline function first_passage_time(
        x,
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

function boundary_condition_ancest(problem, results, approx::FiniteStateApprox, model::CellPopulationModel)
    problem = results.problem

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    fpt(τ) = first_passage_time(results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx)

    return (τ,p) -> 
                sum(Ms .* fpt(τ)) +
                reshape(vec(Ms[end]) .* bndA' * vec(results.cmesol(τ)), tuple(problem.approx.truncation...))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel, method::ToxicBoundaryDeath)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    fpt(τ) = first_passage_time(results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx)

    return (τ,p) -> 2*exp(-results.results[:growth_factor]*τ)*sum(Ms .* fpt(τ))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel, method::ToxicBoundaryRe)
    problem = results.problem
    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)


    fpt(τ) = first_passage_time(results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx)

#    Mprod = vec(Ms[end]) .* vec(bndA)
    return (τ,p) -> 2*exp(-results.results[:growth_factor]*τ)*(
                sum(Ms .* fpt(τ)) +
                reshape(0.5 .* vec(Ms[end]) .* bndA' * vec(results.cmesol(τ)), tuple(problem.approx.truncation...)))
#                0.5 .* reshape(Mprod .* vec(results.cmesol(τ)), tuple(problem.approx.truncation...)))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel, method::Reinsert)
    problem = results.problem

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    bndA = abs.(vec(sum(A, dims=1)))

    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)


    fpt(τ) = first_passage_time(results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx)

  #  Mprod = vec(Ms[end]) .* vec(bndA)
    return (τ,p) -> 2*exp(-results.results[:growth_factor]*τ)*(
                sum(Ms .* fpt(τ)) +
                reshape(vec(Ms[end]) .* bndA' * vec(results.cmesol(τ)), tuple(problem.approx.truncation...)))
#                reshape(Mprod .* vec(results.cmesol(τ)), tuple(problem.approx.truncation...)))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::MotherCellModel, method::Reinsert)
    problem = results.problem

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    #bndA = reshape(bndA, tuple(problem.approx.truncation...))
    bndA = abs.(vec(sum(A, dims=1)))

    axes_ = collect.(axes(results.results[:birth_dist]))
    states = collect(Iterators.product(axes_...)) 
    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)

    fpt(τ) = first_passage_time(results.results[:birth_dist], 
            τ, 
            problem.ps, 
            results.cmesol; 
            model=problem.model,
            approx=problem.approx)

    return (τ,p) -> exp(-results.results[:growth_factor]*τ)*(
                sum(Ms .* fpt(τ)) +
                vec(Ms[end]) .* bndA' * results.cmesol(τ))
end
