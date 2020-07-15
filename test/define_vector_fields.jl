using OrdinaryDiffEq, StaticArrays
const ODE = OrdinaryDiffEq

heaviside(x) = x > 0 ? one(x) : zero(x)

rot_double_gyre = ODE.ODEFunction{false}((u, p, t) -> begin
    SVector{2}(-((2*π*sin(u[1]*π)*cos(2*u[2]*π)
          *(t ^ 2*(3 - 2t)*heaviside(1 - t)
          *heaviside(t) + heaviside(-1 + t))
           + π*cos(u[2]*π)*sin(2*u[1]*π)
          *(1 - (t ^ 2*(3 - 2t)*heaviside(1 - t)
          *heaviside(t) + heaviside(-1 + t))))),
         2*π*sin(u[2]*π)*cos(2*u[1]*π)
          *(1 - (t ^ 2*(3 - 2t)*heaviside(1 - t)
          *heaviside(t) + heaviside(-1 + t)))
           + π*cos(u[1]*π)*sin(2*u[2]*π)
          *(t ^ 2*(3 - 2t)*heaviside(1 - t)
          *heaviside(t) + heaviside(-1 + t)))
    end
)

rot_double_gyre! = ODE.ODEFunction{true}((du, u, p, t) -> begin
    du[1] = -((2*π*sin(u[1]*π)*cos(2*u[2]*π)*(t ^ 2*(3 - 2t)*heaviside(1 - t)*heaviside(t) + heaviside(-1 + t))
           + π*cos(u[2]*π)*sin(2*u[1]*π)*(1 - (t ^ 2*(3 - 2t)*heaviside(1 - t)*heaviside(t) + heaviside(-1 + t)))))
    du[2] = 2*π*sin(u[2]*π)*cos(2*u[1]*π)*(1 - (t ^ 2*(3 - 2t)*heaviside(1 - t)*heaviside(t) + heaviside(-1 + t)))
           + π*cos(u[1]*π)*sin(2*u[2]*π)*(t ^ 2*(3 - 2t)*heaviside(1 - t)*heaviside(t) + heaviside(-1 + t))
    return du
    end
)

bickleyJet = ODE.ODEFunction{false}((u, p, t) -> begin
    SVector{2}(
        -((-0.00012532*(0.0075*cos(0.313922461152095*(u[1] - t*(2.888626e-5 - 1.604096e-5*(-1 + √5)))) + 0.15*cos(0.627844922304191*(-1.28453e-5t + u[1])) + 0.3*cos(0.941767383456286*(-2.888626e-5t + u[1])))*sech(0.564971751412429*u[2]) ^ 2*tanh(0.564971751412429*u[2]) - 6.266e-5*(1 - tanh(0.564971751412429*u[2]) ^ 2))),
        0.0001109082*((-0.00235441845864072*sin(0.313922461152095*(u[1] - t*(2.888626e-5 - 1.604096e-5*(-1 + √5)))) - 0.0941767383456286*sin(0.627844922304191*(-1.28453e-5t + u[1]))) - 0.282530215036886*sin(0.941767383456286*(-2.888626e-5t + u[1])))*sech(0.564971751412429*u[2]) ^ 2)
    end
)

bickleyJet! = ODE.ODEFunction{true}((du, u, p, t) -> begin
    du[1] = -((-0.00012532*(0.0075*cos(0.313922461152095*(u[1] - t*(2.888626e-5 - 1.604096e-5*(-1 + √5)))) + 0.15*cos(0.627844922304191*(-1.28453e-5t + u[1])) + 0.3*cos(0.941767383456286*(-2.888626e-5t + u[1])))*sech(0.564971751412429*u[2]) ^ 2*tanh(0.564971751412429*u[2]) - 6.266e-5*(1 - tanh(0.564971751412429*u[2]) ^ 2)))
    du[2] = 0.0001109082*((-0.00235441845864072*sin(0.313922461152095*(u[1] - t*(2.888626e-5 - 1.604096e-5*(-1 + √5)))) - 0.0941767383456286*sin(0.627844922304191*(-1.28453e-5t + u[1]))) - 0.282530215036886*sin(0.941767383456286*(-2.888626e-5t + u[1])))*sech(0.564971751412429*u[2]) ^ 2
    return du
    end
)


var_rot_double_gyre = begin
          (ODE).ODEFunction{false}(((var"#249#u", var"#250#p", var"#251#t")->begin
                      var"#246#X" = StaticArrays.SMatrix{2, 2}(var"#249#u"[1, 2], var"#249#u"[2, 2], var"#249#u"[1, 3], var"#249#u"[2, 3])
                      var"#247#DV" = StaticArrays.SMatrix{2, 2}(-((2 * π ^ 2 
                                     * cos(var"#249#u"[1, 1] * π) * cos(2 
                                     * var"#249#u"[2, 1] * π) * (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") 
                                     * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) + 0.5 
                                     * (1 + sign(-1 + var"#251#t"))) + 2 * π ^ 2 * cos(
                                     var"#249#u"[2, 1] * π) * cos(2 * var"#249#u"[1, 1] * π) * (1 - (0.25 * var"#251#t" ^ 2 
                                     * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) + 0.5 
                                     * (1 + sign(-1 + var"#251#t")))))), -(π ^ 2) * sin(var"#249#u"[1, 1] * π) * sin(2 * var"#249#u"[2, 1] *
                                     π) * (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) 
                                     + 0.5 * (1 + sign(-1 + var"#251#t"))) - 4 * π ^ 2 * sin(var"#249#u"[2, 1] * π) * sin(2 * var"#249#u"[1, 1] 
                                     * π) * (1 - (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) 
                                     + 0.5 * (1 + sign(-1 + var"#251#t")))), -((-4 * π ^ 2 * sin(var"#249#u"[1, 1] * π) * sin(2 * var"#249#u"[2, 1] * 
                                     π) * (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) 
                                     + 0.5 * (1 + sign(-1 + var"#251#t"))) - π ^ 2 * sin(var"#249#u"[2, 1] * π) * sin(2 * var"#249#u"[1, 1] 
                                     * π) * (1 - (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 
                                     - var"#251#t")) + 0.5 * (1 + sign(-1 + var"#251#t")))))), 2 * π ^ 2 * cos(var"#249#u"[1, 1] * π) 
                                     * cos(2 * var"#249#u"[2, 1] * π) * (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) 
                                     * (1 + sign(1 - var"#251#t")) + 0.5 * (1 + sign(-1 + var"#251#t"))) + 2 * π ^ 2 * cos(var"#249#u"[2, 1] 
                                     * π) * cos(2 * var"#249#u"[1, 1] * π) * (1 - (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") 
                                     * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) + 0.5 * (1 + sign(-1 + var"#251#t")))))
                      var"#248#DX" = var"#247#DV" * var"#246#X"

                      return StaticArrays.SMatrix{2, 3}(-((2 * π * sin(var"#249#u"[1, 1] * π) * cos(2 * var"#249#u"[2, 1] * π) * (0.25 
                                     * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) + 0.5 
                                     * (1 + sign(-1 + var"#251#t"))) + π * cos(var"#249#u"[2, 1] * π) * sin(2 * var"#249#u"[1, 1] * π) 
                                     * (1 - (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 - var"#251#t")) 
                                     + 0.5 * (1 + sign(-1 + var"#251#t")))))), 2 * π * sin(var"#249#u"[2, 1] * π) * cos(2 * var"#249#u"[1, 1] 
                                     * π) * (1 - (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) * (1 + sign(1 
                                     - var"#251#t")) + 0.5 * (1 + sign(-1 + var"#251#t")))) + π * cos(var"#249#u"[1, 1] * π) 
                                     * sin(2 * var"#249#u"[2, 1] * π) * (0.25 * var"#251#t" ^ 2 * (3 - 2var"#251#t") * (1 + sign(var"#251#t")) 
                                     * (1 + sign(1 - var"#251#t")) + 0.5 * (1 + sign(-1 + var"#251#t"))), var"#248#DX"[1, 1], 
                                     var"#248#DX"[2, 1], var"#248#DX"[1, 2], var"#248#DX"[2, 2])
                  end))
      end
