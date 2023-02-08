struct FiniteStateApprox <: AbstractAnalyticalApprox
    truncation::Vector{Int64}
    problem
    A::SparseMatrixCSC
    boundary
    partition
    states
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

        BoundaryConditionWrapper = FunctionWrappers.FunctionWrapper{Any, Tuple{Any, Any}}

        return new(truncation, fsp_problem, A, bndA, Ms, states, divr, fpt)
    end
end

function cmemodel(
        xinit,
        parameters::Vector{Float64},
        tspan::Tuple{Float64, Float64},
        model::AbstractPopulationModel,
        approx::FiniteStateApprox)

    Axu = similar(vec(xinit))
    divy = similar(xinit) 

    function fu!(dx, x, p, τ)
        approx.divr(τ, divy)
        mul!(Axu, approx.A, vec(x))
        dx .= reshape(Axu, size(xinit)) .- divy .* x
    end
    
    return ODEProblem(fu!, xinit, tspan, parameters)
end

function jointmodel(
        λ::Float64,
        xinit,
        parameters::Vector{Float64},
        tspan::Tuple{Float64, Float64},
        model::AbstractPopulationModel,
        approx::FiniteStateApprox)

    Axu = similar(vec(xinit))
    divy = similar(vec(xinit)) 

    function fu!(dx, x, p, τ)
        mul!(Axu, approx.A, vec(x))
        approx.divr(τ, divy)
        dx .= reshape(Axu, size(xinit)) .- divy .* x - λ .* x
    end
    
    return ODEProblem(fu!, xinit, tspan, parameters)
end

function jointinit(
    results,
    model::CellPopulationModel)
    return (2 * results.results[:growth_factor]) .* results.results[:birth_dist]
end

function jointinit(
    results,
    model::MotherCellModel)
    solver = results.solver

    integrand(u,p) = division_time_dist(results)(u)
    prob(s) = IntegralProblem(integrand, s, results.problem.tspan[end], 0.0)
    normal_integrand(u,p) = solve(prob(u), solver.integrator; reltol=solver.rtol, abstol=solver.atol).u
    prob_normal = IntegralProblem(normal_integrand, 0, results.problem.tspan[end], 0.0)
    normal = solve(prob_normal, solver.integrator; reltol=solver.rtol, abstol=solver.atol).u

    return results.results[:birth_dist] ./ normal
end

function error(results, approx::FiniteStateApprox)
    problem = results.problem
    model = results.problem.model

    function ferr!(du, u, p, t)
        du[1] = approx.boundary' * vec(p[3](t)) * exp(-p[1]*t)
    end

    return ODEProblem(ferr!, [0.0,], problem.tspan, [0.0,]) 
end

function first_passage_time(
        x,
        τ::Float64, 
        p::Vector{Float64}, 
        Π,
        ratepre; 
        divr)
    # TODO: in general need to pass in also the reaction network to see which
    # indices correspond to which counts. 
    # First passage time for division.
    divr(τ, ratepre)
    return ratepre .* Π(τ)
end

function boundary_condition_ancest(problem, results, approx::FiniteStateApprox, model::CellPopulationModel)
    problem = results.problem
    return (τ,p) -> 
        sum(approx.partition .* approx.fpt(results.results[:birth_dist], τ, results.cmesol)) +
        reshape(vec(approx.partition[end]) .* approx.boundary' * vec(results.cmesol(τ)), tuple(problem.approx.truncation...))
end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel, method::ToxicBoundaryDeath)
    # Assuming no mass enters is added to the system the boundary states
    # correspond to where the columns of A are negative.
    ratepre = similar(results.results[:birth_dist])
    return (τ,p) -> 2*exp(-results.results[:growth_factor]*τ)*sum(approx.partition .* approx.fpt(results.results[:birth_dist], τ, results.cmesol, ratepre))
end

#function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel, method::ToxicBoundaryRe)
#    problem = results.problem
#    fsp_problem = FSPSystem(model.molecular_model)
#    A = convert(SparseMatrixCSC, fsp_problem, tuple(approx.truncation...), problem.ps, 0)
#    # Assuming no mass enters is added to the system the boundary states
#    # correspond to where the columns of A are negative.
#    bndA = abs.(vec(sum(A, dims=1)))
#
#    axes_ = collect.(axes(results.results[:birth_dist]))
#    states = collect(Iterators.product(axes_...)) 
#    Ms = partition.(problem.model.partition_kernel, states, approx.truncation)
#
#
#    fpt(τ) = first_passage_time(results.results[:birth_dist], 
#            τ, 
#            problem.ps, 
#            results.cmesol; 
#            model=problem.model,
#            approx=problem.approx)
#
#    return (τ,p) -> 2*exp(-results.results[:growth_factor]*τ)*(
#                sum(Ms .* fpt(τ)) +
#                reshape(0.5 .* vec(Ms[end]) .* bndA' * vec(results.cmesol(τ)), tuple(problem.approx.truncation...)))
#end

#function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel, method::Reinsert)
#    problem = results.problem
#    ratepre = similar(results.results[:birth_dist])
#    return (τ,p) -> 2*exp(-results.results[:growth_factor]*τ)*(
#               sum(approx.partition .* approx.fpt(results.results[:birth_dist], τ, results.cmesol, ratepre)) +
#               reshape(vec(approx.partition[end]) .* problem.approx.boundary' * vec(results.cmesol(τ)), tuple(problem.approx.truncation...)))
#end

function boundary_condition(problem, results, approx::FiniteStateApprox, model::CellPopulationModel, method::Reinsert)
    problem = results.problem
    return (τ, p) -> 2*exp(-p[1]*τ)*(
        sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
        reshape(vec(approx.partition[end]) .* problem.approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...)))
end


function boundary_condition(problem, results, approx::FiniteStateApprox, model::MotherCellModel, method::Reinsert)
    problem = results.problem
    return (τ, p) -> sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
        reshape(vec(approx.partition[end]) .* problem.approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...))
end
