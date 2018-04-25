
addprocs()
mkdir("docs/build")
mkdir("docs/build/img/")
@everywhere using CoherentStructures
@everywhere function checkerboard(x)
    return ((-1)^(floor((x[1]*10)%10)))*((-1)^(floor((x[2]*10)%10)))
end

plot_u(ctx2,u2,100,100)
inverse_flow_map_t = (t,u0) -> flow(rot_double_gyre,u0,[t,0.0])[end]

@everywhere ctx2 = regularTriangularGrid((200,200))
@everywhere u2 = nodal_interpolation(ctx2,checkerboard)
@everywhere uf(t) = u2
extra_kwargs_fun = t->  [(:title, @sprintf("Rotating Double Gyre, t=%.2f", t))]
res = CoherentStructures.eulerian_video(ctx2,uf,inverse_flow_map_t,
    0.0,1.0, 500,500,100, [0.0,0.0],[1.0,1.0],colorbar=false,
    extra_kwargs_fun=extra_kwargs_fun)
Plots.mp4(res ,"docs/build/img/rotdoublegyre.mp4")
