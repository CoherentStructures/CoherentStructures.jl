@everywhere using Tensors, DiffEqBase, OrdinaryDiffEq#, ForwardDiff

# this function is intended for potential flow map linearization by automatic differentiation
# @everywhere function flow(g,u0,tspan)
#   Tspan = (tspan[1], tspan[end])
#   prob = ODEProblem(g,u0,eltype(u0).(Tspan))
#   sol = solve(prob,saveat=tspan,save_everystep=false,dense=false,reltol=1e-7,abstol=1e-7).u[end]
# end

@everywhere @inline function arraymap(myfun,howmanytimes::Int64,basesize::Int64,t::Float64,x::Array{Float64},result::Array{Float64})
    @inbounds for i in 1:howmanytimes
        @views @inbounds  myfun(t,x[ 1 + (i-1)*basesize:  i*basesize],result[1 + (i - 1)*basesize: i*basesize])
    end
end

@everywhere function LinearizedFlowMap(odefun,x₀::Vector{Float64},tspan::Vector{Float64},δ::Float64)

    dx = [δ,0.]; dy = [0.,δ];
    stencil = [x₀+dx; x₀+dy; x₀-dx; x₀-dy]
    prob = ODEProblem((t,x,result) -> arraymap(odefun,4,2,t,x,result),stencil,(tspan[1],tspan[end]))
    # Tsit5 seems to be a bit faster than BS5 in this case
    sol = solve(prob,DP5(),saveat=tspan,save_everystep=false,dense=false,reltol=1.e-3,abstol=1.e-3).u
    return [Tensor{2,2}((s[1:4] - s[5:8])./2δ) for s in sol]
end

@everywhere function LinearizedFlowMap(odefun,x₀::Vector{Float64},tspan::Vector{Float64})

    prob = ODEProblem(odefun,[x₀; 1.; 0.; 0.; 1.],(tspan[1],tspan[end]))
    # BS5 seems to be a bit faster than Tsit5 in this case
    sol = solve(prob,DP5(),saveat=tspan,save_everystep=false,dense=false,reltol=1e-3,abstol=1e-3).u
    return [Tensor{2,2}(reshape(s[3:6],2,2)) for s in sol]
end

@everywhere function PullBackTensors(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,2,Float64,3})

    G = inv(D)
    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    DFinv = inv.(DF)
    DF = [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
    DFinv = [symmetric(df⋅(D⋅transpose(df))) for df in DFinv]
    return DF, DFinv # DF is pullback metric tensor, DFinv is pullback diffusion tensor
end

@everywhere function PullBackMetricTensor(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,G::Tensors.SymmetricTensor{2,2,Float64,3})

    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    return [symmetric(transpose(df)⋅(G⋅df)) for df in DF]
end

@everywhere function PullBackDiffTensor(odefun,x₀::Vector{Float64},tspan::Vector{Float64},
    δ::Float64,D::Tensors.SymmetricTensor{2,2,Float64,3})

    iszero(δ) ? DF = LinearizedFlowMap(odefun,x₀,tspan) : DF = LinearizedFlowMap(odefun,x₀,tspan,δ)
    DF = inv.(DF)
    return [symmetric(df⋅(D⋅transpose(df))) for df in DF]
end
