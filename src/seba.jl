#(c) 2018 Nathanael Schilling
# Implement SEBA (see https://web.maths.unsw.edu.au/~froyland/FRS18a.pdf)

"""
    SEBA(V; [μ=0.99/sqrt(size(V,1)), tol=1e-14, maxiter=1000])

Implements the SEBA algorithm (see https://web.maths.unsw.edu.au/~froyland/FRS18a.pdf).

"""
function SEBA(Vin; μ=0.99/sqrt(size(Vin, 1)), tol=1e-14, maxiter=5000)
    V = Matrix(qr(Vin).Q)
    S = zero(Vin)
    R = diagm(0 => ones(size(Vin, 2)))#TODO: Allow this as optional argument?
    for vi in eachcol(V)
        if abs(-(extrema(vi)...)) < 1e-14
            vi .+= (rand.() - 0.5) * 1e-12
        end
    end
    numiter = 1
    #R = zeros(size(R))
    Z = similar(Vin)
    SV = similar(R)
    Rnew = similar(R)
    while true
        mul!(Z, V, R')
        for (si, zi) in zip(eachcol(S), eachcol(Z))
            si .= (sign.(zi) .* max.(abs.(zi) .- μ, 0))
            colNorm = norm(si)
            if colNorm != 0
                si ./= colNorm
            end
        end
        mul!(SV, S', V)
        svdres = svd!(SV)
        mul!(Rnew, svdres.U, svdres.V')
        if opnorm(Rnew - R) < tol
            break
        elseif numiter > maxiter
            throw(error("numiter > $maxiter"))
        end
        R = Rnew
        numiter += 1
    end

    for si in eachcol(S)
        si .*= sign(sum(si))
        si ./= maximum(si)
    end

    return S[:, sortperm([-minimum(si) for si in eachcol(S)])]
end
