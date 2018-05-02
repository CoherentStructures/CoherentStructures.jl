include("expression_substitutions.jl")

sgn(x) = (x > 0) ? one(x) : (x < 0) ? -one(x) : zero(x)
heaviside(x) = 0 < x ? one(x) : zero(x)


# manually define  derivatives for functions that SymEngine cant differentiate
diff_dict = Dict()
diff_dict[:abs] = :sgn
diff_dict[:sgn] = :zero
diff_dict[:heaviside] = :zero
diff_dict[:zero] = :zero

# not a nice way to differentiate expressions, but ReverseDiffSource
# is broken.
function expr_diff(expr::Expr, var::Symbol)
    expr_sym = SymEngine.Basic(expr)
    d_expr_sym = SymEngine.diff(expr_sym, var)

    d_expr = parse(SymEngine.toString.(d_expr_sym))

    # clean up derivatives that SymEngine didn't differentiate
    d_expr = clumsy_cleaner(d_expr)

    # clean up unresolved substitutions
    d_expr = clumsy_cleaner2(d_expr)

    # clean up zeros
    d_expr = clumsy_simplifier(d_expr)
end
expr_diff(expr, var::Symbol) = expr == var ? 1 : 0


function clumsy_cleaner(expr::Expr)
# some functions like abs(x) are not treated by SymEngine and block the
# expression. For example, diff(Basic(:(abs(x^2+1)),:x)) returns a SymEngine Object
# whose string representation is parsed to :(Derivative(abs(1 + x ^ 2), x)),
# whose AST is:
# Expr
#  head: Symbol call
#  args: Array{Any}((3,))
#    1: Symbol Derivative
#    2: Expr
#      head: Symbol call
#      args: Array{Any}((2,))
#        1: Symbol abs
#        2: Expr
#          head: Symbol call
#          args: Array{Any}((3,))
#            1: Symbol +
#            2: Int64 1
#            3: Expr
#              head: Symbol call
#              args: Array{Any}((3,))
#                1: Symbol ^
#                2: Symbol x
#                3: Int64 2
#              typ: Any
#          typ: Any
#      typ: Any
#    3: Symbol x
# typ: Any

# detect expressions of this form
    if expr.head == :call && expr.args[1] == :Derivative
        f =  expr.args[2].args[1]
        var = expr.args[3]
        f_arg = expr.args[2].args[2]
        if haskey(diff_dict, f)  # try if diff_dict provides a rescue
            df = diff_dict[f]
            inner_d = expr_diff(f_arg, var)
            df_computed_manually = :($df($f_arg)*$inner_d)
            return clumsy_cleaner(df_computed_manually)       # handle nested problems
        end
    end
    return Expr(expr.head, clumsy_cleaner.(expr.args)...) # call recursively on subexpressions
end
clumsy_cleaner(expr) = expr

# A second thing that SymEngine does is returning expressions of the form
# Subs(ex1, symb1, ex2). Resolve these substitutions
function clumsy_cleaner2(expr::Expr)
    if expr.head == :call && expr.args[1] == :Subs
        return clumsy_cleaner2(sym_subst(expr.args[2], expr.args[3], expr.args[4]))
    end
    return Expr(expr.head, clumsy_cleaner2.(expr.args)...)
end
clumsy_cleaner2(expr) = expr

function clumsy_simplifier(expr::Expr)
    args = clumsy_simplifier.(expr.args)
    if expr.head!=:call
        return Expr(expr.head, args...)
    end
    if args[1] == :zero return 0 end

    if args[1] == :(+)
        args = [arg for arg=args if arg != 0]
        return  Expr(expr.head, args...)
    end
    if args[1] == :(*)
        if any(args[2:end] .== 0) return 0 end
    end

    if args[1] == :(/)
        if args[2] == 0 return 0 end
    end

    if args[1] == :(-)
        if length(args) == 2 return args[2] == 0 ? 0 : Expr(expr.head, args...) end
        if args[2] == 0 return :(-$(args[3])) end
        if args[3] == 0 return :( $(args[2])) end
    end
    Expr(expr.head, args...)
end

clumsy_simplifier(expr) = expr
