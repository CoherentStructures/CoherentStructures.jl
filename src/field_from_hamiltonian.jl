import SymEngine
include("expression_substitutions.jl")
include("expr_diff.jl")

"""makefields(name, :from, name_of_H, def_of_H)

Define a time dependent vector field from stream function. As a definition of
H, provide a begin ... end - block that contains one line `name_of_H = <some_term>`.
`makefields` tries to substitute all occurences of variables in  <some_term> that
are not equal to :x,:y or :t with substitution rules that are defined in the block.
The final expression for the hamiltonian is read as a time dependent scalar field
on `R^2` i.e. x is interpreted as ``x_1``, y as ``x_2`` and `t` as the time.

The macro returns a dictionary with different versions of the field as elements.
``
#Example
fields  = @makefields from H begin
    some_fun(r) = sin(r^2+1)
    r_sq = x^2+y^2
    H = sin(some_fun(r_sq))
end
# get equation of variation
f = fields[:(dU, U, p, t)]
"""
macro makefields(keyword::Symbol, Hamiltonian::Symbol, code::Expr)
    @assert keyword == :from "unknow word \"$keyword\", should be \"from\""

    # collect all the info in <code> to get expression for the Hamiltonian
    H = symbolic_substitutions(code, Hamiltonian, [:x,:y,:t, keys(diff_dict)...])

    # get symbolic derivatives
     ∇H  = expr_diff.( H, [:x,:y])
    ∇²H  = expr_diff.(∇H, [:x :y])

    # final expressions for components of vector field F and its derivative
    # to prepend a minus to an expression we use :(-$ex)
    F  = [:(-$(∇H[2])), ∇H[1]]
    DF = [:(-$(∇²H[2,1])) :(-$(∇²H[1,1]));
               ∇²H[2,2]        ∇²H[1,2] ]

   # substitute occurences of :x and :y by access of array elements.
   # We use the expressions for F only when u is a 2x1-Vector.
   # DF is only used in eq_var, where u has additional
   # columns indicating the current differential DΦ of the
   # flow map.
     F = sym_subst.( F, [[:x,:y]], [[:(u[1]), :(u[2])]])
    DF = sym_subst.(DF, [[:x,:y]], [[:(u[1,1]), :(u[2,1])]])



    # final code: define a Dictionary that contains the field in different
    # formats. Accessible by function "signatures"
    quote
        output = Dict()

        function field!(du, u, p, t)
            du[1] = $(F[1])
            du[2] = $(F[2])
            du
        end

        # save f ∈ R^2x1 and Df ∈ R^2x2 in a 2x3-matrix
        function eq_var(du, u, p, t)
            # write field into first column
            field!(du[:,1], u[:,1], p, t)

            # store DF in a Tensor here to (hopefully) make it a bit faster
            dF = Tensor{2,2}(($(DF[1,1]), $(DF[2,1]),
                              $(DF[1,2]), $(DF[2,2])))

            # store the matrix part of the Equation of Variation
            du[:,2:3] = dF ⋅ Tensor{2,2}(u[:,2:3])
            du
        end

        # define functions with different signatures
        field(u, p, t) = field!(zeros(T,2), [x, y], p, t)
        field2(x::T,y::T,t::T) where T = field([x,y], nothing, t)
        field_at_t(t)       = (x,y) -> field(x, y, t)

        # populate the output dictionary
        output[:(u,t)]      = field
        output[:(du,u,p,t)] = field!
        output[:(dU,U,p,t)] = eq_var

        output[:(t)]        = field_at_t
        output[:(x,y,t)]    = field2


        # bind the dictionary to the name that was passed to the macro
        output
    end
end


f = fields[:(dU, U, p, t)]
