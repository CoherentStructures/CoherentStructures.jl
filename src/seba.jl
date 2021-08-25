#(c) 2018-2021 Nathanael Schilling, maintained by Daniel Karrasch

"""
    SEBA(V, R = diagm(0 => ones(size(Vin, 2))); [μ=0.99/sqrt(size(V,1)), tol=1e-14, maxiter=1000])

Computes a "sparse eigenbasis approximation" (SEBA) as proposed in [1].

`V` is the matrix containing the eigenvectors as columns, `R` is an optional initial
rotation matrix. For the role of the keyword arguments, see ref. [1].

[1] Froyland, G. and Rock, Chr. P. and Sakellariou, K. [Sparse eigenbasis approximation:
Multiple feature extraction across spatiotemporal scales with application to coherent set
identification](https://doi.org/10.1016/j.cnsns.2019.04.012). _Communications in Nonlinear
Science and Numerical Simulation_, 77:81-107, 2019. 
"""
function SEBA(Vin, R = diagm(0 => ones(size(Vin, 2))); μ=0.99/sqrt(size(Vin, 1)), tol=1e-14, maxiter=5000)
    V = Matrix(qr(Vin).Q)
    for vi in eachcol(V)
        mini, maxi = extrema(vi)
        if maxi - mini < 1e-14
            vi .+= (rand(size(Vin, 1)) .- 1//2) .* 1e-12
        end
    end
    numiter = 1
    S = similar(Vin)
    SV = similar(R)
    Rnew = similar(R)
    while true
        mul!(S, V, transpose(R))
        for si in eachcol(S)
            si .= (sign.(si) .* max.(abs.(si) .- μ, 0))
            colNorm = norm(si)
            if !iszero(colNorm)
                si ./= colNorm
            end
        end
        mul!(SV, S', V)
        svdres = svd!(SV)
        mul!(Rnew, svdres.U, svdres.Vt)
        if opnorm(Rnew - R) < tol
            break
        elseif numiter > maxiter
            throw(error("numiter > maxiter = $maxiter"))
        end
        R, Rnew = Rnew, R
        numiter += 1
    end

    for si in eachcol(S)
        si .*= sign(sum(si))
        si ./= maximum(si)
    end

    return S[:, sortperm([-minimum(si) for si in eachcol(S)])]
end
