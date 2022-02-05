struct CellPopulationModel <: AbstractPopulationModel
    molecular_model::ReactionSystem
    division_rate_symbolic
    partition_kernel::AbstractPartitionKernel
    division_rate::Function

    function CellPopulationModel(molecular_model, division_rate_symbolic, partition_kernel)
        _division_rate = gen_division_rate_function(division_rate_symbolic, molecular_model) 
        new(molecular_model, division_rate_symbolic, partition_kernel, _division_rate)
    end
end
