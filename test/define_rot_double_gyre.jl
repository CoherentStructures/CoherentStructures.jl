using OrdinaryDiffEq
const ODE = OrdinaryDiffEq

heaviside(x) = x > 0 ? one(x) : zero(x)
rot_double_gyre = (ODE).ODEFunction{false}(((var"#25#u", var"#26#p", var"#27#t")->begin
SVector{2}(-((2 * π * sin(var"#25#u"[1] * π) * cos(2 * var"#25#u"[2] * π) 
          * (var"#27#t" ^ 2 * (3 - 2var"#27#t") * heaviside(1 - var"#27#t") 
          * heaviside(var"#27#t") + heaviside(-1 + var"#27#t")) 
          + π * cos(var"#25#u"[2] * π) * sin(2 * var"#25#u"[1] * π) 
          * (1 - (var"#27#t" ^ 2 * (3 - 2var"#27#t") * heaviside(1 - var"#27#t") 
          * heaviside(var"#27#t") + heaviside(-1 + var"#27#t"))))), 
        2 * π * sin(var"#25#u"[2] * π) * cos(2 * var"#25#u"[1] * π) 
          * (1 - (var"#27#t" ^ 2 * (3 - 2var"#27#t") * heaviside(1 - var"#27#t") 
          * heaviside(var"#27#t") + heaviside(-1 + var"#27#t"))) 
          + π * cos(var"#25#u"[1] * π) * sin(2 * var"#25#u"[2] * π) 
          * (var"#27#t" ^ 2 * (3 - 2var"#27#t") * heaviside(1 - var"#27#t")
          * heaviside(var"#27#t") + heaviside(-1 + var"#27#t")))
end))




rot_double_gyre_vareq =  (ODE).ODEFunction{false}(((var"#32#u", var"#33#p", var"#34#t")->begin
                var"#29#X" = SMatrix{2, 2}(var"#32#u"[1, 2], var"#32#u"[2, 2], var"#32#u"[1, 3], var"#32#u"[2, 3])
                var"#30#DV" = SMatrix{2, 2}(-((2 * π ^ 2 * cos(var"#32#u"[1, 1] * π) 
                              * cos(2 * var"#32#u"[2, 1] * π) * (var"#34#t" ^ 2 * (3 - 2var"#34#t") 
                              * heaviside(1 - var"#34#t") * heaviside(var"#34#t") + heaviside(-1 + var"#34#t")) 
                              + 2 * π ^ 2 * cos(var"#32#u"[2, 1] * π) * cos(2 * var"#32#u"[1, 1] * π) 
                              * (1 - (var"#34#t" ^ 2 * (3 - 2var"#34#t") * heaviside(1 - var"#34#t") 
                              * heaviside(var"#34#t") + heaviside(-1 + var"#34#t"))))), -(π ^ 2) 
                              * sin(var"#32#u"[1, 1] * π) * sin(2 * var"#32#u"[2, 1] * π) * (var"#34#t" ^ 2 
                              * (3 - 2var"#34#t") * heaviside(1 - var"#34#t") * heaviside(var"#34#t") 
                              + heaviside(-1 + var"#34#t")) - 4 * π ^ 2 * sin(var"#32#u"[2, 1] * π) 
                              * sin(2 * var"#32#u"[1, 1] * π) * (1 - (var"#34#t" ^ 2 * (3 - 2var"#34#t") 
                              * heaviside(1 - var"#34#t") * heaviside(var"#34#t") + heaviside(-1 + var"#34#t"))), 
                              -((-4 * π ^ 2 * sin(var"#32#u"[1, 1] * π) * sin(2 * var"#32#u"[2, 1] * π) 
                              * (var"#34#t" ^ 2 * (3 - 2var"#34#t") * heaviside(1 - var"#34#t") * heaviside(var"#34#t") 
                              + heaviside(-1 + var"#34#t")) - π ^ 2 * sin(var"#32#u"[2, 1] * π) 
                              * sin(2 * var"#32#u"[1, 1] * π) * (1 - (var"#34#t" ^ 2 * (3 - 2var"#34#t") 
                              * heaviside(1 - var"#34#t") * heaviside(var"#34#t") + heaviside(-1 + var"#34#t"))))),
                              2 * π ^ 2 * cos(var"#32#u"[1, 1] * π) * cos(2 * var"#32#u"[2, 1] * π) 
                              * (var"#34#t" ^ 2 * (3 - 2var"#34#t") * heaviside(1 - var"#34#t") 
                              * heaviside(var"#34#t") + heaviside(-1 + var"#34#t")) + 2 * π ^ 2 * cos(var"#32#u"[2, 1] * π) 
                              * cos(2 * var"#32#u"[1, 1] * π) * (1 - (var"#34#t" ^ 2 * (3 - 2var"#34#t") 
                              * heaviside(1 - var"#34#t") * heaviside(var"#34#t") + heaviside(-1 + var"#34#t"))))
                var"#31#DX" = var"#30#DV" * var"#29#X"
                return SMatrix{2, 3}(-((2 * π * sin(var"#32#u"[1, 1] * π) * cos(2 * var"#32#u"[2, 1] * π) 
                                    * (var"#34#t" ^ 2 * (3 - 2var"#34#t") * heaviside(1 - var"#34#t") * heaviside(var"#34#t") 
                                    + heaviside(-1 + var"#34#t")) + π * cos(var"#32#u"[2, 1] * π) 
                                    * sin(2 * var"#32#u"[1, 1] * π) * (1 - (var"#34#t" ^ 2 * (3 - 2var"#34#t") 
                                    * heaviside(1 - var"#34#t") * heaviside(var"#34#t") + heaviside(-1 + var"#34#t"))))), 
                                    2 * π * sin(var"#32#u"[2, 1] * π) * cos(2 * var"#32#u"[1, 1] * π) 
                                    * (1 - (var"#34#t" ^ 2 * (3 - 2var"#34#t") * heaviside(1 - var"#34#t") 
                                    * heaviside(var"#34#t") + heaviside(-1 + var"#34#t"))) + π * cos(var"#32#u"[1, 1] * π) 
                                    * sin(2 * var"#32#u"[2, 1] * π) * (var"#34#t" ^ 2 * (3 - 2var"#34#t") 
                                    * heaviside(1 - var"#34#t") * heaviside(var"#34#t") + heaviside(-1 + var"#34#t")), 
                                    var"#31#DX"[1, 1], var"#31#DX"[2, 1], var"#31#DX"[1, 2], var"#31#DX"[2, 2])
        end))


rot_double_gyre! = ODE.ODEFunction{true}((du, u, p, t) -> du .= rot_double_gyre(u, p, t))
