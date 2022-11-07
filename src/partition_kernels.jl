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
    x_ = x .- ones(Int, length(truncation))
    out_ = prod.(collect(Iterators.product(pdf.(Binomial.(x_, kernel.parameter))...)))
    return PaddedView(0.0, out_, tuple(truncation...))
end

partition(kernel::AbstractPartitionKernel, x::Int64, truncation::Int64) = _partition(kernel, x, truncation)
Broadcast.broadcasted(::typeof(partition), kernel, x, truncation) = 
    broadcast(_partition, Ref(kernel), x, Ref(truncation))
