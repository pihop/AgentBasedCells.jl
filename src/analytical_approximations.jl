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

