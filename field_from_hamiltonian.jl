import SymEngine
include("expression_substitutions.jl")
include("expr_diff.jl")

"""makefields(name, :from, name_of_H, def_of_h)

Define a time dependent vector field from stream function
#Example
@makefields fields from H begin
    some_fun(r) = sin(r^2+1)
    r_sq = x^2+y^2
    H = sin(r_sqr)
end
"""
macro makefields(name::Symbol, keyword::Symbol, Hamiltonian::Symbol, code::Expr)
    @assert keyword == :from "unknow word \"$keyword\", should be \"from\""

    println("Getting Expression for $Hamiltonian")

    # compile code into an expression for the Hamiltonian
    stream_expr = symbolic_substitutions(code, Hamiltonian, [:x,:y,:t, keys(diff_dict)...])

    # get symbolic derivatives
    gradient_expr = expr_diff.(stream_expr, [:x,:y])
    hessian_expr  = expr_diff.(gradient_expr, [:x :y])

    # make functions directly acces array elements
    gradient_expr = sym_subst.(gradient_expr, [:x], [:(u[1])])
    gradient_expr = sym_subst.(gradient_expr, [:y], [:(u[2])])
    hessian_expr = sym_subst.(hessian_expr, [:x], [:(u[1])])
    hessian_expr = sym_subst.(hessian_expr, [:y], [:(u[2])])


    ###
    # final code: define a Dictionary that contains the field in different
    # formats. Accessible by function "signatures"
    quote
        println("Executing Code")
        output = Dict()

        # defin
        function field!(v, u, t)
            v[1] = -$(gradient_expr[2])
            v[2] = $(gradient_expr[1])
            v
        end

        # save f (2x1) and Df (2x2) in a 2x3-matrix
        function var_eq(t, u, du)
            du[1,1] = -$(gradient_expr[2])
            du[2,1] = $(gradient_expr[1])
            du[1,2] = -$(hessian_expr[2,1])
            du[2,2] = $(hessian_expr[2,2])
            du[1,3] = -$(hessian_expr[1,1])
            du[2,3] = $(hessian_expr[1,2])
            du
        end

        # define functions different signatures
        field(x::T,y::T,t::T) where T = field!(zeros(T,2), [x, y], t)
        field_at_t(t) = (x,y) -> field(x,y,t)
        f_prealloc(t,u,du) = field!(du, u, t)
        f_noprealloc(t,u)  = field(u[1], u[2], t)

        # populate the output dictionary
        output[:(t)]        = field_at_t
        output[:(x,y,t)]    = field
        output[:(v!,x,y,t)] = field!
        output[:(t,u,du)]   = f_prealloc
        output[:(t,u)]      = f_noprealloc
        output[:(t,U,DU)]   = var_eq

        # bind the dictionary to the
        $(esc(name)) = output
    end
end
