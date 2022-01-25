@kwdef struct PopulationExperimentSetup <: AbstractExperimentSetup
    init::Vector{Float64} 
    model_parameters::Vector{Float64} 
    tspan_simulation::Tuple{Float64, Float64} 
    tspan_analytical::Tuple{Float64, Float64} 
    Δt::Float64 
    max_pop::Int64 
    truncation::Vector{Int64} 
    jitt::Float64 = 1e-4
    molecular_model_rn 
    division_rate
end

