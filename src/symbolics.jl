function _gen_division_rate_function(symb_rate, rn::ReactionSystem)
    return @RuntimeGeneratedFunction build_function(symb_rate, states(rn), Catalyst.parameters(rn), ModelingToolkit.get_iv(rn);
        conv = ModelingToolkit.states_to_sym(states(rn)),
        expression=Val{true})
end

gen_division_rate_function(x, rn::ReactionSystem) = 
    _gen_division_rate_function(x, rn)
Broadcast.broadcasted(::typeof(gen_division_rate_function), x, rn) = 
    broadcast(_gen_division_rate_function, x, Ref(rn))

