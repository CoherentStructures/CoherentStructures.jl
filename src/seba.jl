#(c) 2018 Nathanael Schilling
# Implement SEBA (see https://web.maths.unsw.edu.au/~froyland/FRS18a.pdf)

"""
    SEBA(V; [μ=0.99/sqrt(size(V,1)), tol=1e-14, maxiter=1000])

Implements the SEBA algorithm (see https://web.maths.unsw.edu.au/~froyland/FRS18a.pdf).

"""
function SEBA(Vin; μ=0.99/sqrt(size(Vin, 1)), tol=1e-14, maxiter=5000)
    n, nev = size(Vin)
    V = Matrix(qr(Vin).Q)
    S = zeros(n, nev)
    R = diagm(0 => ones(size(Vin, 2)))#TODO: Allow this as optional argument?
    for i in 1:nev
        if maximum(V[:,i]) - minimum(V[:,i]) < 1e-14
            V[:,i] .+= (rand() - 0.5) * 1e-12
        end
    end
    numiter = 1
    #R = zeros(size(R))
    while true
        Z = V * permutedims(R)
        for i in 1:nev
            S[:,i] .= (sign.(Z[:,i]) .* max.(abs.(Z[:,i]) .- μ, 0))
            colNorm = norm(S[:,i])
            if colNorm != 0
                S[:,i] ./= colNorm
            end
        end
        svdres = svd(S'V)
        Rnew = svdres.U * svdres.V'
        if opnorm(Rnew - R) < tol
            R = Rnew
            break
        elseif numiter > maxiter
            throw(AssertionError("numiter > $maxiter"))
        end
        R .= Rnew
        numiter += 1
    end

    for i in 1:nev
        S[:,i] .*= sign.(sum(S[:,i]))
        S[:,i] ./= maximum(S[:,i])
    end

    return S[:, sortperm([-minimum(S[:,i]) for i in 1:nev])]
end
