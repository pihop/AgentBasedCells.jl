abstract type AbstractPartitionKernel end

struct BinomialKernel <: AbstractPartitionKernel
    parameter::Float64
end

function _partition(kernel::BinomialKernel, x::Int64, truncation::Vector{Int64})
    out = zeros(prod(truncation))    
    out[1:x+1] = pdf(Binomial(x, kernel.parameter))
    return out
end

partition(kernel::AbstractPartitionKernel, x::Int64, truncation::Int64) = _partition(kernel, x, truncation)
Broadcast.broadcasted(::typeof(partition), kernel, x, truncation) = 
    broadcast(_partition, Ref(kernel), x, Ref(truncation))
