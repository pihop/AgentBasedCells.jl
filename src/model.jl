abstract type AbstractDivisionRate end

struct DivisionRateMonotonicInc <: AbstractDivisionRate
    γ
    ratef::Function
    function DivisionRateMonotonicInc(γ, rn)
        ratef = gen_division_rate_function(γ, rn)  
        return new(γ, ratef)
    end
end

function divisionrate(u,p,t,rate::DivisionRateMonotonicInc)
    return rate.ratef.(u,Ref(p),Ref(t))
end

function get_λmax(rate::DivisionRateMonotonicInc, traj, tspan, ps)
    return divisionrate(traj(tspan[end]), ps, tspan[end], rate)[end]
end


struct DivisionRateBounded <: AbstractDivisionRate
    γ
    ratef::Function
    ratemax::Function
    function DivisionRateBounded(γ, γmax, rn)
        ratef = gen_division_rate_function(γ, rn)  
        ratemax = gen_division_rate_function(γmax, rn)  
        return new(γ, ratef, ratemax)
    end
end

function divisionrate(u,p,t,rate::DivisionRateBounded)
    return rate.ratef.(u,Ref(p),Ref(t))
end

function divisionratebnd(u,p,t,rate::DivisionRateBounded)
    return rate.ratemax(u,p,t)
end

function get_λmax(rate::DivisionRateBounded, traj, tspan, ps)
    return divisionratebnd(traj(tspan[1]), ps, tspan[1], rate)
end

struct CellPopulationModel <: AbstractPopulationModel
    molecular_model::ReactionSystem
    division_rate::AbstractDivisionRate
    partition_kernel::AbstractPartitionKernel

    function CellPopulationModel(molecular_model, division_rate, partition_kernel; )
        new(molecular_model, division_rate, partition_kernel)
    end
end

function Base.show(io::IO, ::CellPopulationModel)
    print(io, "Cell population model.")
end

struct MotherCellModel <: AbstractPopulationModel
    molecular_model::ReactionSystem
    division_rate::AbstractDivisionRate
    partition_kernel::AbstractPartitionKernel

    function MotherCellModel(molecular_model, division_rate, partition_kernel; )
        new(molecular_model, division_rate, partition_kernel)
    end
end

function Base.show(io::IO, ::MotherCellModel)
    print(io, "Mother cell model.")
end

