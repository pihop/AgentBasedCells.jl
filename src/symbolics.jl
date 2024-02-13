RuntimeGeneratedFunctions.init(@__MODULE__)

function gen_division_rate_function(symb_rate, rn::ReactionSystem)
    return Symbolics.build_function(
        symb_rate, 
        states(rn), 
        Catalyst.parameters(rn), 
        ModelingToolkit.get_iv(rn);
        force_SA = true,
#        conv = ModelingToolkit.states_to_sym(states(rn)),
        expression=Val{false})
end

_gen_division_rate_function(x, rn::ReactionSystem) = gen_division_rate_function(x, rn)
Broadcast.broadcasted(::typeof(gen_division_rate_function), x, rn) = 
    broadcast(_gen_division_rate_function, x, Ref(rn))

