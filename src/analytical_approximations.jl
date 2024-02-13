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
#        A = create_sparsematrix(fsp_problem, tuple(truncation...), ps, 0)
        # Assuming no mass enters is added to the system the boundary states
        # correspond to where the columns of A are negative.
        bndA = abs.(vec(sum(A, dims=1))) 
        
        axes_ = collect.(Base.OneTo.(truncation))
        states = collect(Iterators.product(axes_...)) 
        states = map(x -> x .- tuple(I), states)
        Ms = ArrayOfSimilarArrays(partition.(model.partition_kernel, states, truncation))

        RateWrapper = FunctionWrapper{Nothing, Tuple{Real, Any}}
        divr = RateWrapper((t, dest) -> divisionrate(states, ps, t, dest, model.division_rate))

        FPTWrapper = FunctionWrapper{Any, Tuple{Any, Any, Any, Any}}
        fpt = FPTWrapper((u,t,cmesol,ratepre) -> first_passage_time(u, t, ps, cmesol, ratepre; divr=divr))

        return new{length(truncation)}(truncation, A, reshape(bndA, truncation...), Ms, divr, fpt)
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
        nothing
    end
    
    prob = ODEProblem(fu!, xinit, tspan, 
        (parameters, 
         PreallocationTools.dualcache(similar(vec(xinit))), 
         PreallocationTools.dualcache(similar(xinit)));)
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
        nothing
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
        nothing
    end

    prob = ODEProblem(fu!, xinit, tspan, 
        (parameters, 
         PreallocationTools.dualcache(similar(vec(xinit))), 
         PreallocationTools.dualcache(similar(xinit))))
    return prob
end


function jointinit(results, ::CellPopulationModel)
    return (2 * results.results[:growth_rate]) .* results.results[:birth_dist]
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
        rate_ = sum(approx.boundary .* p[3](t))
        du[1] = rate_ * 2^m * exp(-p[1]*t)
        du[2] = rate_ 
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

#function boundary_condition_ancest(problem, results, approx::FiniteStateApprox, ::Union{CellPopulationModel, MotherCellModel})
#    problem = results.problem
#    return (τ, p) -> 
#    sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
#        reshape(vec(approx.partition[end]) .* approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...))
#end

function elementwise_mat!(ret, A, B)  
    Threads.@threads for i in eachindex(ret) 
    #@inbounds @fastmath @simd for i in eachindex(ret) 
        @inbounds @fastmath begin 
            ret[i] .= A[i] .* B[i]
        end
    end
    nothing
end

function elementwise_mat!(ret, A, B, C)  
    Threads.@threads for i in eachindex(ret) 
        @inbounds @fastmath begin 
        ret[i] .= A[i] .* B[i] .* C[i]
        end
    end
    nothing
end

function boundary_condition(problem, results, approx::FiniteStateApprox, ::CellPopulationModel, ::Reinsert, integrator)
    problem = results.problem
    Tmax = problem.tspan[end]
    a ⊕ b = a .+= b

    divisionsA_ = similar(approx.partition)
    divisions_ = similar(approx.partition[1])

    bnd_stateA_ = similar(approx.partition)
    bnd_state_ = similar(approx.partition[1])

    bnd_timeA_ = similar(approx.partition)
    bnd_time_ = similar(approx.partition[1])

    function f(τ::Float64, p)
        elementwise_mat!(divisionsA_, approx.partition, approx.fpt(p[2], τ, p[3], p[4]))
        divisions_ .= reduce(⊕, view(divisionsA_,2:length(divisionsA_)), init=copy(divisionsA_[1]));

        elementwise_mat!(bnd_stateA_, approx.partition, approx.boundary, p[3](τ))
        bnd_state_ = reduce(⊕, view(bnd_stateA_,2:length(bnd_stateA_)), init=copy(bnd_stateA_[1]));

        elementwise_mat!(bnd_timeA_, approx.partition, p[3](Tmax))
        bnd_time_ = reduce(⊕, view(bnd_timeA_,2:length(bnd_timeA_)), init=copy(bnd_timeA_[1]));

        return 2 * exp(-p[1] * τ) .* (divisions_ .+ 0.5 .* bnd_state_) .+ 1/Tmax * exp(-p[1] * Tmax) .* bnd_time_
    end
    return f
end

#function boundary_condition(problem, results, approx::FiniteStateApprox, ::CellPopulationModel, ::Divide)
#    problem = results.problem
#    return (τ, p) -> 2*exp(-p[1]*τ)*(
#        sum(approx.partition .* approx.fpt(p[2], τ, p[3], p[4])) +
#        reshape(vec(approx.partition[end]) .* approx.boundary' * vec(p[3](τ)), tuple(problem.approx.truncation...)))
#end

function boundary_condition(problem, results, approx::FiniteStateApprox, ::MotherCellModel, ::Reinsert, integrator)
    problem = results.problem
    Tmax = problem.tspan[end]
    a ⊕ b = a .+= b

    divisionsA_ = similar(approx.partition)
    divisions_ = similar(approx.partition[1])

    bnd_stateA_ = similar(approx.partition)
    bnd_state_ = similar(approx.partition[1])

    bnd_timeA_ = similar(approx.partition)
    bnd_time_ = similar(approx.partition[1])

    function f(τ::Float64, p)
        elementwise_mat!(divisionsA_, approx.partition, approx.fpt(p[2], τ, p[3], p[4]))
        divisions_ .= reduce(⊕, view(divisionsA_,2:length(divisionsA_)), init=copy(divisionsA_[1]));

        elementwise_mat!(bnd_stateA_, approx.partition, approx.boundary, p[3](τ))
        bnd_state_ .= reduce(⊕, view(bnd_stateA_,2:length(bnd_stateA_)), init=copy(bnd_stateA_[1]));

        elementwise_mat!(bnd_timeA_, approx.partition, p[3](Tmax))
        bnd_time_ .= reduce(⊕, view(bnd_timeA_,2:length(bnd_timeA_)), init=copy(bnd_timeA_[1]));

        return divisions_ .+ bnd_state_ .+ 1/Tmax .* bnd_time_
    end
    return f
end
