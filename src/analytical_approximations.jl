struct FiniteStateApprox{N} <: AbstractAnalyticalApprox
    truncation::NTuple{N, Int64}
    A
    boundary
    partition
    divr
    fpt
    function FiniteStateApprox(truncation, model, ps)
        fsp_problem = FSPSystem(model.molecular_model)

        A = convert(SparseMatrixCSC, fsp_problem, tuple(truncation...), ps, 0)
        # Assuming no mass enters is added to the system the boundary states
        # correspond to where the columns of A are negative.
        bndA = abs.(vec(sum(A, dims=1)))

        axes_ = collect.(Base.OneTo.(truncation))
        states = collect(Iterators.product(axes_...)) 
        states = map(x -> x .- tuple(I), states)
        Ms = partition.(model.partition_kernel, states, truncation)

        RateWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Float64, Any}}
        divr = RateWrapper((t, dest) -> divisionrate(states, ps, t, dest, model.division_rate))

        FPTWrapper = FunctionWrappers.FunctionWrapper{Any, Tuple{Any, Any, Any, Any}}
        fpt = FPTWrapper((u,t,cmesol,ratepre) -> first_passage_time(u, t, ps, cmesol, ratepre; divr=divr))

        return new{length(truncation)}(truncation, A, bndA, Ms, divr, fpt)
    end
end

function cmemodel(
        xinit,
        parameters::NTuple{N, Float64},
        tspan::Tuple{Float64, Float64},
        model::AbstractPopulationModel,
        approx::FiniteStateApprox) where {N} 

    function fu!(dx, x, (p, Axu, divy), τ)
        Axu = get_tmp(Axu, first(x)*τ)
        divy = get_tmp(divy, first(x)*τ)

        approx.divr(τ, divy)
        mul!(Axu, approx.A, vec(x))
        dx .= reshape(Axu, size(xinit)) .- divy .* x
    end
    
    prob = ODEProblem(fu!, xinit, tspan, 
        (parameters, 
         PreallocationTools.dualcache(similar(vec(xinit))), 
         PreallocationTools.dualcache(similar(xinit))))
    return prob
end

function jointmodel(
        xinit,
        λ::Float64,
        parameters::NTuple{N, Float64},
        tspan::Tuple{Float64, Float64},
        model::AbstractPopulationModel,
        approx::FiniteStateApprox) where {N}

    function fu!(dx, x, (p, Axu, divy), τ)
        Axu = get_tmp(Axu, first(x)*τ)
        divy = get_tmp(divy, first(x)*τ)

        approx.divr(τ, divy)
        mul!(Axu, approx.A, vec(x))
        dx .= reshape(Axu, size(xinit)) .- divy .* x - λ .* x
    end
    
    prob = ODEProblem(fu!, xinit, tspan, 
        (parameters, 
         PreallocationTools.dualcache(similar(vec(xinit))), 
         PreallocationTools.dualcache(similar(xinit))))
    return prob
end

function backwardjointmodel(
    xinit,
    λ::Float64,
    parameters::NTuple{N, Float64},
    tspan::Tuple{Float64, Float64},
    model::AbstractPopulationModel,
    approx::FiniteStateApprox) where {N}

    function fu!(dx, x, (p, xuA, divy), τ)
        xuA = get_tmp(xuA, first(x)*τ)
        divy = get_tmp(divy, first(x)*τ)

        mul!(xuA', vec(x)', approx.A)
        approx.divr(tspan[end] - τ, divy)
        dx .= reshape(xuA', size(xinit)) .- divy .* x - λ .* x
    end

    prob = ODEProblem(fu!, xinit, tspan, 
        (parameters, 
         PreallocationTools.dualcache(similar(vec(xinit))), 
         PreallocationTools.dualcache(similar(xinit))))
    return prob
end


function jointinit(results, ::CellPopulationModel)
    return (2 * results.results[:growth_factor]) .* results.results[:birth_dist]
end

function jointinit(results, ::MotherCellModel)
    solver = results.solver
    return results.results[:birth_dist]# ./ normal
end

function error(results, approx::FiniteStateApprox)
    problem = results.problem

    m = 1
    if typeof(results.problem.model) == MotherCellModel
        m = 0
    end

    function ferr!(du, u, p, t)
        du[1] = approx.boundary' * vec(p[3](t)) * 2^m * exp(-p[1]*t)
        du[2] = approx.boundary' * vec(p[3](t)) 
    end

    return ODEProblem(ferr!, [0.0, 0.0], problem.tspan, [0.0,]) 
end

function first_passage_time(
        x,
        τ::Float64, 
        p::NTuple{N, Float64}, 
        Π,
        ratepre; 
        divr) where {N}
    # TODO: in general need to pass in also the reaction network to see which
    # indices correspond to which counts. 
    # First passage time for division.
    divr(τ, ratepre)
    return ratepre .* Π(τ)
end

function boundary_condition_ancest(problem, results, approx::FiniteStateApprox, ::Union{CellPopulationModel, MotherCellModel})
    problem = results.problem
    return (τ, p) -> 
    sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
        reshape(vec(approx.partition[end]) .* approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, ::CellPopulationModel, ::Reinsert)
    problem = results.problem
    Tmax = problem.tspan[end]
    return (τ, p) -> 2*exp(-p[1]*τ)*(
        sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
        0.5 * reshape(vec(approx.partition[end]) .* approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...))) + 
        1/Tmax*exp(-p[1]*Tmax)*reshape(vec(approx.partition[end]) .* vec(p[3](Tmax)), tuple(problem.approx.truncation...))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, ::CellPopulationModel, ::Divide)
    problem = results.problem
    return (τ, p) -> 2*exp(-p[1]*τ)*(
        sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
        reshape(vec(approx.partition[end]) .* approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...)))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, ::MotherCellModel, ::Reinsert)
    problem = results.problem
    Tmax = problem.tspan[end]
    return (τ, p) -> sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
        reshape(vec(approx.partition[end]) .* approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...)) + 
        1/Tmax * reshape(vec(approx.partition[end]) .* sum(vec(p[3](Tmax))), tuple(problem.approx.truncation...))
end
