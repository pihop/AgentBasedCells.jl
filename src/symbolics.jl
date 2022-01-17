function gen_division_rate_function(symb_rate::Num, rn)
    return @RuntimeGeneratedFunction build_function(symb_rate, states(rn), parameters(rn), ModelingToolkit.get_iv(rn);
        conv = ModelingToolkit.states_to_sym(states(rn)),
        expression=Val{true})
end
