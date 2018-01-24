""" perform all substitutions that are defined in `code` until
    the resulting expression does not contain free variables.
    variables can be bound by `knowns`
"""
symbolic_substitutions(code::Expr, variable::Symbol, knowns = []) = begin
    ex = quote $variable end
    maxit = 20; count = 0
    while has_free_symb(ex,knowns) && (count < maxit)
        ex = compile_expression(code, ex)
        ex = remove_blocks(Base.remove_linenums!(ex))
        count = count + 1
    end
    if has_free_symb(ex, knowns)
        warn("$(remove_blocks(ex)) still has free variables that are not bound by $knowns")
    end
    return ex
end


""" perform all substitutions that are defined in `code` once
"""
compile_expression(defn::Expr, target::Expr) = begin
    if defn.head == :(=)
        f_sig = signature(defn.args[1])
        f_body = defn.args[2]
        ret  = call_subst(target, f_sig, f_body)
        return ret
    end
    compiler(part_comp, ex) = compile_expression(ex, part_comp)
    reduce(compiler, target, defn.args)
end
compile_expression(defn::Expr, target::Symbol) = begin
    compile_expression(defn, Base.remove_linenums!(quote begin $target end end))
end
compile_expression(defn, target) = target


""" substitute all function calls of f in expr
"""
call_subst(expr::Expr, f_sig, f_body) = begin
    if expr.head == :call && f_sig[1] == expr.args[1]
        @assert length(f_sig) == length(expr.args) "$(expr.args) has wrong number of args for $(f_sig)"
        re_expr = f_body
        for (k,sym) = enumerate(f_sig[2:end])
            re_expr = sym_subst(re_expr, sym, expr.args[k+1])
        end
        return re_expr
    end
    Expr(expr.head, call_subst.(expr.args, [f_sig], [f_body])...)
end
call_subst(expr, f_sign, f_body) =  begin
    if expr == f_sign[1]
        @assert length(f_sign) == 1 "wrong number of arguments"
        return f_body
    end
    expr
end

""" sym_subst(expr, sym, s_expr)

replace all occurences of `sym` in `expr` by `s_expr`
"""
sym_subst(expr::Expr, sym, s_expr) = begin
    Expr(expr.head, sym_subst.(expr.args, [sym], [s_expr])...)
end
sym_subst(expr::Symbol, sym, s_expr) = expr==sym ? s_expr: expr
sym_subst(expr, sym, s_expr) = expr

""" does <ex> contain a symbol that is not bound by <bound_vars>?
"""
has_free_symb(ex::Expr, bound_vars) = begin
    !all((!).(has_free_symb.(ex.args, [bound_vars])))
end
has_free_symb(ex::Symbol, bound_vars) = begin
    !(contains(==, bound_vars, ex) | isdefined(Base, ex))
end
has_free_symb(ex, bound_vars) = false

""" clean up enclosing blocks to get to the core expression
"""
remove_blocks(ex::Expr) = begin
    if ex.head == :block
        return remove_blocks(ex.args[1])
    else
        return Expr(ex.head, remove_blocks.(ex.args)...)
    end
end
remove_blocks(ex) = ex

""" get signature [<f_name> <arg1> <arg2> ...], where a symbol is a function without arguments
"""
signature(ex::Symbol) = [ex]
signature(ex::Expr) = ex.args
