include("../src/field_from_hamiltonian.jl")

# after this, bickley will reference a Dictionary of functions
# access it via the desired signature. e.g. F = bickley[:(dU, U, p, t)]
# for the right side of the equation of variation.

bickley = @makefields from stream begin
    stream = psi₀ + psi₁
    psi₀   = - U₀ * L₀ * tanh(y / L₀)
    psi₁   =   U₀ * L₀ * sech(y / L₀)^2 * re_sum_term

    re_sum_term =  Σ₁ + Σ₂ + Σ₃

    Σ₁  =  ε₁ * cos(k₁*x - K₁*c₁*t)
    Σ₂  =  ε₂ * cos(k₂*x - K₂*c₂*t)
    Σ₃  =  ε₃ * cos(k₃*x - K₃*c₃*t)

    k₁ = 2/r₀     ; k₂ = 4/r₀    ; k₃ = 6/r₀

    lx  = 6.371e6π ; ly = 1.777e6

    K₁ = 2π / lx  ; K₂ = 4π / lx ; K₃ = 6π / lx
    ε₁ = 0.0075  ; ε₂ = 0.15    ; ε₃ = 0.3
    c₁ = 0.1446U₀ ; c₂ = 0.205U₀ ; c₃ = 0.461U₀

    U₀ = 62.66e-6; L₀ = 1770e-3; r₀ = 6371e-3
end
