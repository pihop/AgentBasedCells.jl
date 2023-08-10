mutable struct CellState 
    # Keeps track of the cell state and cell age.
    state::Array{Float64}
    τ::Float64 
    birth_time::Float64
    birth_state::Array{Float64}
    division_time::Float64
    division_sampler::NonHomogeneousSampling
    sim
    function CellState(state, τ, birth_time, birth_state, division_time, sampler)
        cellstate = new()
        cellstate.state = state
        cellstate.τ = τ
        cellstate.birth_time = birth_time
        cellstate.birth_state = birth_state
        cellstate.division_time = division_time
        cellstate.division_sampler = sampler
        cellstate.sim = nothing
        return cellstate
    end
end

function Base.show(io::IO, ::CellState)
    print(io, "Structure for cell state.")
end

function molecule_state(cell_state::CellState)
    return cell_state.state
end

function birth_molecule_state(cell_state::CellState)
    return cell_state.birth_state
end

function cell_size(cell_state::CellState)
    return sum(cell_state.state)
end

function cell_birth_size(cell_state::CellState)
    return sum(cell_state.birth_state)
end

function cell_age(cell_state::CellState)
    return cell_state.τ
end

function cell_division_time(cell_state::CellState)
    return cell_state.division_time
end

function cell_birth_time(cell_state::CellState)
    return cell_state.birth_time
end

function cell_division_age(cell_state::CellState)
    return cell_state.division_time - cell_state.birth_time
end

function state_at_division(cell_state::CellState)
    return cell_state.sim(cell_division_age(cell_state))
end

function state_at_times(cell_state::CellState, tstep)
    maxt = cell_division_age(cell_state)
    ts_ = 0.0:tstep:maxt

    if mod(maxt, tstep) > 0.5 * tstep 
        return [getindex.(cell_state.sim(vcat(ts_, maxt)).u, 1)..., cell_state.sim(maxt)[1]]
    else
        return getindex.(cell_state.sim(vcat(ts_, maxt)).u, 1)
    end
end
