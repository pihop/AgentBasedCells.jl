struct FiniteStateApprox <: AbstractAnalyticalApprox
    truncation::Vector{Int64}
    tspan::Tuple{Float64, Float64}
end

function cmemodel(
        xinit::Vector{Float64},
        parameters::Vector{Float64},
        model::CellPopulationModel,
        approximation::FiniteStateApprox)

    fsp_problem = FSPSystem(model.molecular_model)
    A = convert(SparseMatrixCSC, fsp_problem, tuple(approximation.truncation...), parameters, 0)

    states = CartesianIndices(zeros(approximation.truncation...))
    states = map(x -> x.I .- tuple(I), states)

    function fu!(dx, x, p, τ)
        dx[:] = A * x - model.division_rate.(states, fill(p, size(states)), τ) .* x
    end

    return ODEProblem(fu!, xinit, approximation.tspan, parameters)
end
