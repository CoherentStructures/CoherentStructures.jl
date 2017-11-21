include("FEMDL.jl")

# flow map
function rot_double_gyre(t,x,dx)
  st = ((t>0)&(t<1))*t^2*(3-2*t) + (t>=1)*1
  dxΨP = 2π*cos.(2π*x[:,1]).*sin.(π*x[:,2])
  dyΨP = π*sin.(2π*x[:,1]).*cos.(π*x[:,2])
  dxΨF = π*cos.(π*x[:,1]).*sin.(2π*x[:,2])
  dyΨF = 2π*sin.(π*x[:,1]).*cos.(2π*x[:,2])
  dx[:,1] = - ((1-st)dyΨP + st*dyΨF)
  dx[:,2] = (1-st)dxΨP + st*dxΨF
end
tspan = (0.0,1.0)
steps = 2
Φ(x) = flow_map(rot_double_gyre,x,tspan,steps)

# data points
nx = 50; n = nx^2
x0 = rand(n,2)
#scatter(x0[:,1],x0[:,2],markersize=.1)
pb = [1:n 1:n]

# time integration and triangulation
@time p = Φ(x0)
#scatter!(p[2][:,1],p[2][:,2],color="red",markersize=.1)

n = size(x0,1)
t1, tri = delaunay(p[1])
t2, tri = delaunay(p[2])
#triplot(tri)

G = kron([1 0 0 1],ones(size(t1,1)))
D1,M = assemble(p[1],t1,pb,G)
G = kron([1 0 0 1],ones(size(t2,1)))
D2,M = assemble(p[2],t2,pb,G)
@time (λ,ev,nconv,niter,nmult,resid) = eigs(0.5*(D1+D2),M;nev=3,which=:SM)
@show λ, nconv, niter, nmult, norm(resid)
ev
plot(contour(reshape(ev[:,3],nx,nx),fill=true))
