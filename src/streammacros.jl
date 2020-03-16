# (c) 2018 Alvaro de Diego

stream_dict = Dict()

"""
    @define_stream(name::Symbol, code::Expr)

Define a scalar stream function on ``R^2``. The defining code can be a series of
definitions in an enclosing `begin ... end`-block and is treated as a series of
symbolic substitutions. Starting from the symbol `name`, substitutions are
performed until the resulting expression only depends on `x`, `y` and `t`.

The symbol `name` is not brought into the namespace. To access the resulting
vector field and variational equation  use `@velo_from_stream name` and
`@var_velo_from_stream name`

This is a convenience macro for the case where you want to use
`@velo_from_stream` and `@var_velo_from_stream` without typing the code twice.
If you only use one, you might as well use `@velo_from_stream name code` or
`@var_velo_from_stream` directly.
"""
macro define_stream(name::Symbol, code::Expr)
    haskey(
        stream_dict,
        name,
    ) && (@warn "overwriting definition of stream $name")
    stream_dict[name] = code
    quote
        # do nothing, the actual work is done when @velo_from_stream name
        # is called.
    end
end

"""
    @velo_from_stream(name::Symbol, [code::Expr])

Get the velocity field corresponding to a stream function on ``R^2``. The defining
code can be a series of definitions (in an enclosing `begin ... end`-block and
is treated as a series of symbolic substitutions. Starting from the symbol
`name`, substitutions are performed until the resulting expression only depends
on `x`, `y` and `t`.

The macro returns an anonymous function with signature `(u,p,t)` that returns an
`SVector{2}` corresponding to the vector field at position `u` at time `t`. The
parameter slot is not used and can be filled with `nothing` when calling.

The macro can be called without the `code` if the stream `name` has been defined
beforehand via `@define_stream`.

!!! note "Sign convention"
    We follow the "oceanographic" sign convention, whereby the velocity ``v``
    is derived from the stream function ``\\psi`` by
    ``v = (-\\partial_y\\psi, \\partial_x\\psi).``

### Examples
```
julia> using CoherentStructures

julia> f = @velo_from_stream Ψ_ellipse begin
               Ψ_ellipse = a*x^2 + b*y^2
               a = t
               b = 3
           end
(#3) #1 (generic function with 1 method)

julia> f([1.0,1.0], nothing, 1.0)
2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:
 -6.0
  2.0
```
```
julia> using CoherentStructures

julia> @define_stream Ψ_circular begin
           Ψ_circular = f(x) + g(y)
           # naming of function variables
           # does not matter:
           f(a) = a^2
           g(y) = y^2
       end

julia> f2 = @velo_from_stream Ψ_circular
(#5) #1 (generic function with 1 method)

julia> f2([1.0,1.0], nothing, 0.0)
2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:
 -2.0
  2.0
```
"""
macro velo_from_stream(H::Symbol, formulas::Expr)
    V, _ = streamline_derivatives(H, formulas)
    V = sym_subst.(V, [[:x, :y, :p]], [[:(u[1]), :(u[2]), :(p[1])]])
    quote
        ODE.ODEFunction{false}((u, p, t) -> SVector{2}($(V[1]), $(V[2])))
    end
end
macro velo_from_stream(name::Symbol)
    haskey(stream_dict, name) || (@error "stream $name not defined")
    quote
        @velo_from_stream $name $(stream_dict[name])
    end
end

"""
    @var_velo_from_stream(name::Symbol, [code::Expr])

Get the (state and tangent space) velocity field corresponding to a stream
function on ``R^2``. The defining code can be a series of definitions (in an
enclosing `begin ... end`-block and is treated as a series of symbolic
substitutions. Starting from the symbol `name`, substitutions are performed
until the resulting expression only depends on `x`, `y` and `t`.

The macro returns an anonymous function with signature `(U,p,t)` that returns an
`SMatrix{2,3}`: in the first column, one has the usual velocity, in the second
to third column, one has the linearized velocity, both at position `u = U[:,1]`
at time `t`. The parameter slot is not used and can be filled with `nothing`
when calling.

The macro can be called without the `code` if the stream `name` has been defined
beforehand via `@define_stream`.

!!! note "Sign convention"
    We follow the "oceanographic" sign convention, whereby the velocity ``v``
    is derived from the stream function ``\\psi`` by
    ``v = (-\\partial_y\\psi, \\partial_x\\psi).``
"""
macro var_velo_from_stream(H::Symbol, formulas::Expr)
    V, DV = streamline_derivatives(H, formulas)

    # substitute :x and :y by access to the first column of a matrix u
    V = sym_subst.(V, [[:x, :y, :p]], [[:(u[1, 1]), :(u[2, 1]), :(p[1])]])
    DV = sym_subst.(DV, [[:x, :y, :p]], [[:(u[1, 1]), :(u[2, 1]), :(p[1])]])

    quote
        ODE.ODEFunction{false}(
            (u, p, t) -> begin
                # take as input a  2x3 matrix which is interpreted the following way:
                # [ x[1] X[1,1] X[1,2]
                #   x[2] X[2,1] X[2,2] ]

                # current
                X = SMatrix{2,2}(u[1, 2], u[2, 2], u[1, 3], u[2, 3])
                DV = SMatrix{2,2}(
                    $(DV[1, 1]),
                    $(DV[2, 1]),
                    $(DV[1, 2]),
                    $(DV[2, 2]),
                )
                DX = DV * X
                return SMatrix{2,3}(
                    $(V[1]),
                    $(V[2]),
                    DX[1, 1],
                    DX[2, 1],
                    DX[1, 2],
                    DX[2, 2],
                )
            end,
        )
    end
end
macro var_velo_from_stream(name::Symbol)
    haskey(stream_dict, name) || (@error "stream $name not defined")
    quote
        @var_velo_from_stream $(name) $(stream_dict[name])
    end
end

"""
    @vorticity_from_stream(name::Symbol, [code::Expr])

Get the vorticity field as a function of `(x, y, t)` corresponding to a stream
function on ``R^2``.

!!! note "Sign convention"
    The vorticity ``\\omega`` of the velocity field ``v = (v_x, v_y)`` is defined
    as derived from the stream function ``\\psi`` by ``\\omega = \\partial_x v_x -
    \\partial_y v_y) = trace(\\nabla^2\\psi)``, i.e., the trace of the Hessian of
    the stream function.
"""
macro vorticity_from_stream(H::Symbol, formulas::Expr)
    _, DV = streamline_derivatives(H, formulas)

    quote
        (x, y, t) -> $(DV[2, 1]) - $(DV[1, 2])
    end
end
macro vorticity_from_stream(name::Symbol)
    haskey(stream_dict, name) || (@error "stream $name not defined")
    quote
        @vorticity_from_stream $(name) $(stream_dict[name])
    end
end

function streamline_derivatives(H::Symbol, formulas::Expr)
    # symbols that are not supposed to be substituted
    # (additional to symbols defined in Base)
    bound_symbols = [:x, :y, :p, :t, keys(diff_dict)...]
    H = substitutions(H, formulas, bound_symbols)

    # symbolic gradient and hessian (note the broadcast)
    ∇H = expr_diff.([H], [:x, :y])
    ∇²H = expr_diff.(∇H, [:x :y])

    # formula for streamlines (perpendicular to gradient)
    V = [:(-$(∇H[2])), ∇H[1]]

    # equation of variation for streamlines
    DV = [
        :(-$(∇²H[2, 1])) :(-$(∇²H[2, 2]))
        ∇²H[1, 1] ∇²H[1, 2]
    ]

    return V, DV
end

########################################################################################
#            symbolic differentiation of expressions using SymEngine                   #
########################################################################################

sgn(x) = (x > 0) ? one(x) : (x < 0) ? -one(x) : zero(x)
heaviside(x) = 0 < x ? one(x) : zero(x)
window(x, a, b) = (a <= x < b) ? one(x) : zero(x)

# manually define  derivatives for functions that SymEngine cant differentiate
diff_dict = Dict()
diff_dict[:abs] = :sgn
diff_dict[:sgn] = :zero
diff_dict[:heaviside] = :zero
diff_dict[:zero] = :zero
diff_dict[:window] = :zero

function expr_diff(expr::Expr, var::Symbol)
    # not a nice way to differentiate expressions, but ReverseDiffSource
    # is broken.
    expr_sym = SymEngine.Basic(expr)
    d_expr_sym = SymEngine.diff(expr_sym, var)
    d_expr = Meta.parse(SymEngine.toString.(d_expr_sym))

    # resolve derivatives that SymEngine doesn't know using diff_dict
    d_expr = additional_derivatives(d_expr)

    # clean up unresolved substitutions that result from SymEnginge treating
    # unknown derivatives
    d_expr = substitution_cleaner(d_expr)

    # clean up zeros
    d_expr = simple_simplifier(d_expr)
end
expr_diff(expr, var::Symbol) = expr == var ? 1 : 0

function additional_derivatives(expr::Expr)
    # some functions like abs(x) are not treated by SymEngine and block the
    # expression. For example, diff(Basic(:(abs(x^2+1)),:x)) returns a SymEngine Object
    # whose string representation is parsed to :(Derivative(abs(1 + x ^ 2), x)),
    # whose AST is:
    # Expr
    #  head: Symbol call
    #  args: Array{Any}((3,))
    #    1: Symbol Derivative
    #    2: Expr
    #           ... Body of expression ...
    #    3: Symbol x
    # typ: Any

    # detect expressions of this form
    if expr.head === :call && expr.args[1] === :Derivative
        f = expr.args[2].args[1]
        var = expr.args[3]
        f_arg = expr.args[2].args[2]
        if haskey(diff_dict, f)  # try if diff_dict provides a rescue
            df = diff_dict[f]
            inner_d = expr_diff(f_arg, var)
            df_computed_manually = :($df($f_arg) * $inner_d)
            return additional_derivatives(df_computed_manually)       # handle nested problems
        end
    end
    return Expr(expr.head, additional_derivatives.(expr.args)...) # call recursively on subexpressions
end
additional_derivatives(expr) = expr

# A second thing that SymEngine does is returning expressions of the form
# Subs(ex1, symb1, ex2). Resolve these substitutions
function substitution_cleaner(expr::Expr)
    if expr.head === :call && expr.args[1] === :Subs
        return substitution_cleaner(sym_subst(
            expr.args[2],
            expr.args[3],
            expr.args[4],
        ))
    end
    return Expr(expr.head, substitution_cleaner.(expr.args)...)
end
substitution_cleaner(expr) = expr

# perform some basic simplifications like getting rid of ones and zeros
function simple_simplifier(expr::Expr)
    args = simple_simplifier.(expr.args)
    if expr.head !== :call
        return Expr(expr.head, args...)
    end
    if args[1] === :zero
        return 0
    end

    if args[1] === :(+)
        args = [arg for arg = args if arg != 0]
        return Expr(expr.head, args...)
    end
    if args[1] === :(*)
        if any(args[2:end] .== 0)
            return 0
        end
    end

    if args[1] === :(/)
        if args[2] == 0
            return 0
        end
    end

    if args[1] === :(-)
        if length(args) == 2
            return args[2] == 0 ? 0 : Expr(expr.head, args...)
        end
        if args[2] == 0
            return :(-$(args[3]))
        end
        if args[3] == 0
            return :($(args[2]))
        end
    end
    return Expr(expr.head, args...)
end

simple_simplifier(expr) = expr
expr_grad(expr, coord_vars::Vector{Symbol}) = expr_diff.(expr, coord_vars)

function hessian(expr, coord_vars)
    ∇expr = expr_grad(expr, coord_vars)
    ∇²expr = expr_grad(expr)
end

####t####################################################################################
#                 Functions for symbolic manipulation of expressions                    #
#                      (mainly substitutions of function calls)                         #
#########################################################################################

"""
    substitutions(code::Expr, variable::Symbol, knowns = [])

Perform all substitutions that are defined in `code` until
the resulting expression does not contain free variables.
Variables can be bound by `knowns`.
"""
function substitutions(variable::Symbol, code::Expr, knowns = [])
    Base.remove_linenums!(code)
    ex = quote
        $variable
    end
    maxit = 20
    count = 0

    # dumb approach: keep blindly performing substitutions until there are no free
    # variables left
    while has_free_symb(ex, knowns) && (count < maxit)
        ex = substitute_once(code, ex)
        ex = remove_blocks(Base.remove_linenums!(ex))
        count = count + 1
    end
    if has_free_symb(ex, knowns)
        @warn "$(remove_blocks(ex)) still has free variables that are not bound by $knowns"
    end
    return ex
end

"""
    substitute_once(defns::Expr, target::Expr)

Perform all substitutions that are defined in `code` once.
"""
function substitute_once(defns::Expr, target::Expr)
    if defns.head === :(=)
        f_sig = signature(defns.args[1])
        f_body = defns.args[2]
        ret = call_subst(target, f_sig, f_body)
        return ret
    end
    performer(part_comp, ex) = substitute_once(ex, part_comp)
    return reduce(performer, defns.args, init = target)
end
function substitute_once(defns::Expr, target::Symbol)
    return substitute_once(defns, Base.remove_linenums!(quote
        begin
            $target
        end
    end))
end
substitute_once(defns, target) = target

"""
    call_subst(expr::Expr, f_sig, f_body)

Substitute all function calls of `f` in `expr`.
"""
function call_subst(expr::Expr, f_sig, f_body)
    if expr.head === :call && f_sig[1] == expr.args[1]
        @assert length(f_sig) == length(expr.args) "$(expr.args) has wrong number of args for $(f_sig)"
        re_expr = f_body
        for (k, sym) in enumerate(f_sig[2:end])
            re_expr = sym_subst(re_expr, sym, expr.args[k+1])
        end
        return re_expr
    end
    return Expr(expr.head, call_subst.(expr.args, [f_sig], [f_body])...)
end
function call_subst(expr, f_sign, f_body)
    if expr == f_sign[1]
        @assert length(f_sign) == 1 "wrong number of arguments"
        return f_body
    end
    return expr
end

"""
    sym_subst(expr, sym, s_expr)

Replace all occurences of `sym` in `expr` by `s_expr`.
"""
function sym_subst(expr::Symbol, sym::Symbol, s_expr)
    expr == sym ? s_expr : expr
end
function sym_subst(expr::Expr, sym::Symbol, s_expr)
    return Expr(expr.head, sym_subst.(expr.args, [sym], [s_expr])...)
end
function sym_subst(expr, symbols::Vector{Symbol}, bodies::Vector{Expr})
    @assert length(symbols) == length(bodies) "lists have different lengths"
    for (symb, s_expr) in zip(symbols, bodies)
        expr = sym_subst(expr, symb, s_expr)
    end
    return expr
end
# fallback
sym_subst(expr, sym::Symbol, s_expr) = expr

"""
    has_free_symb(ex::Expr, bound_vars)

Does `ex` contain a symbol that is not bound by `bound_vars`?
"""
function has_free_symb(ex::Expr, bound_vars)
    return !all((!).(has_free_symb.(ex.args, [bound_vars])))
end
function has_free_symb(ex::Symbol, bound_vars)
    return !(any(y -> (ex == y), bound_vars) | isdefined(Base, ex))
end
has_free_symb(ex, bound_vars) = false

"""
    remove_blocks(ex::Expr)

Clean up enclosing blocks to get to the core expression.
"""
function remove_blocks(ex::Expr)
    if ex.head === :block
        return remove_blocks(ex.args[1])
    else
        return Expr(ex.head, remove_blocks.(ex.args)...)
    end
end
remove_blocks(ex) = ex

"""
    signature [<f_name> <arg1> <arg2> ...]

A symbol is interpreted as a function without arguments.
"""
signature(ex::Symbol) = [ex]
signature(ex::Expr) = ex.args
