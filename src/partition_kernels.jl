abstract type AbstractPartitionKernel end

struct BinomialKernel <: AbstractPartitionKernel
    parameter::Float64
end

function _partition(kernel::BinomialKernel, x::Int64, truncation::Int64)
    out = zeros(truncation)    
    out[1:x+1] = pdf(Binomial(x, kernel.parameter))
    return out
end

function _partition(kernel::BinomialKernel, x, truncation)
    out_ = prod.(collect(Iterators.product(pdf.(Binomial.(x, kernel.parameter))...)))
    return PaddedView(0.0, out_, tuple(truncation...))
end

function partition_cell(::Type{CellPopulationModel}, kernel::BinomialKernel, cell::CellState, birth_time::Float64)
    # Parition and return the Array of cells.    
    partition = [] 
    partition_ = [] 
    for s in cell.state
        molecule_number = rand(Binomial(Int(s), kernel.parameter))
        push!(partition, molecule_number)
        push!(partition_, Int(s) - molecule_number)  
    end

    return [CellState(partition, 0.0, birth_time, partition, 0.0, cell.idx, ThinningSampler()), 
            CellState(partition_, 0.0, birth_time, partition_, 0.0, cell.idx, ThinningSampler())]
end

function partition_cell(::Type{MotherCellModel}, kernel::BinomialKernel, cell::CellState, birth_time::Float64)
    # Parition and return the Array of cells.    
    partition = [] 
    for s in cell.state
        molecule_number = rand(Binomial(Int(s), kernel.parameter))
        push!(partition, molecule_number)
    end

    return [CellState(partition, 0.0, birth_time, partition, 0.0, cell.idx, ThinningSampler()), ]
end



partition(kernel::AbstractPartitionKernel, x::Int64, truncation::Int64) = _partition(kernel, x, truncation)
Broadcast.broadcasted(::typeof(partition), kernel, x, truncation) = 
    broadcast(_partition, Ref(kernel), x, Ref(truncation))


struct BinomialWithDuplicate <: AbstractPartitionKernel
    parameter::Float64
    idxs::Vector{Int64}
end

function _partition(kernel::BinomialWithDuplicate, x, truncation)
#    x_ = x .- ones(Int, length(truncation))
    probs = []
    for (i, el) in enumerate(x)
        if !in(i, kernel.idxs)
            push!(probs, pdf(Binomial(Int64(el), kernel.parameter)))
        else
            zs = zeros(Float64, el+1)
            zs[end] = 1.0
            push!(probs, zs) 
        end
    end
    out_ = prod.(collect(Iterators.product(probs...)))
    return PaddedView(0.0, out_, tuple(truncation...))
end

function partition_cell(::Type{CellPopulationModel}, kernel::BinomialWithDuplicate, cell::CellState, birth_time::Float64)
    # Parition and return the Array of cells.    
    partition = [] 
    partition_ = [] 
    for (i,s) in enumerate(cell.state)
        if !in(i, kernel.idxs)
            molecule_number = rand(Binomial(Int(s), kernel.parameter))
            push!(partition, molecule_number)
            push!(partition_, Int(s) - molecule_number)  
        else
            push!(partition, s)
            push!(partition_, s)
        end
    end

    return [CellState(partition, 0.0, birth_time, partition, 0.0, cell.idx, ThinningSampler()), 
            CellState(partition_, 0.0, birth_time, partition_, 0.0, cell.idx, ThinningSampler())]
end

function partition_cell(::Type{MotherCellModel}, kernel::BinomialWithDuplicate, cell::CellState, birth_time::Float64)
    # Parition and return the Array of cells.    
    partition = [] 
    for (i,s) in enumerate(cell.state)
        if !in(i, kernel.idxs)
            molecule_number = rand(Binomial(Int(s), kernel.parameter))
            push!(partition, molecule_number)
        else
            push!(partition, s)
        end
    end

    return [CellState(partition, 0.0, birth_time, partition, 0.0, cell.idx, ThinningSampler()), ]
end

struct ConcentrationKernel <: AbstractPartitionKernel
end

function _partition(kernel::ConcentrationKernel, x, truncation)
    probs = []
    for el in x
        zs = zeros(Float64, el+1)
        zs[end] = 1.0
        push!(probs, zs) 
    end
    out_ = prod.(collect(Iterators.product(probs...)))
    return PaddedView(0.0, out_, tuple(truncation...))
end

function partition_cell(::Type{CellPopulationModel}, kernel::ConcentrationKernel, cell::CellState, birth_time::Float64)
    # Parition and return the Array of cells.    
    partition = [] 
    partition_ = [] 
    for s in cell.state
        push!(partition, s)
        push!(partition_, s)
    end

    return [CellState(partition, 0.0, birth_time, partition, 0.0, cell.idx, ThinningSampler()), 
            CellState(partition_, 0.0, birth_time, partition_, 0.0, cell.idx, ThinningSampler())]
end

function partition_cell(::Type{MotherCellModel}, kernel::ConcentrationKernel, cell::CellState, birth_time::Float64)
    # Parition and return the Array of cells.    
    partition = [] 
    for s in cell.state
        push!(partition, s)
    end

    return [CellState(partition, 0.0, birth_time, partition, 0.0, cell.idx, ThinningSampler()), ]
end

