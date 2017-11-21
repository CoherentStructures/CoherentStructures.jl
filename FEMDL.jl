using DifferentialEquations
using Plots
using VoronoiDelaunay

function flow_map(v,x,tspan,steps)
  prob = ODEProblem(v,x,tspan)
  sol = solve(prob, abstol = 1e-3, reltol = 1e-3)
  sol(linspace(tspan[1],tspan[2],steps)).u
end

struct NumberedPoint2D <: AbstractPoint2D
    _x::Float64
    _y::Float64
    _k::Int64
    NumberedPoint2D(x::Float64,y::Float64,k::Int64) = new(x,y,k)
    NumberedPoint2D(x::Float64,y::Float64) = new(x, y, 0)
    NumberedPoint2D(p::Point2D) = new(p.x, p.y, 0)
 end

importall GeometricalPredicates
Point(x::Real, y::Real, k::Int64) = NumberedPoint2D(x, y, k)
Point2D(p::NumberedPoint2D) = Point2D(p._x,p._y)
getx(p::NumberedPoint2D) = p._x
gety(p::NumberedPoint2D) = p._y

function delaunay(x)
  n = size(x,1)
  tess = DelaunayTessellation2D{NumberedPoint2D}(n)
  width = max_coord - min_coord
  a = NumberedPoint2D[Point(min_coord+x[i,1]*width,min_coord+x[i,2]*width,i) for i in 1:n]
  push!(tess,a)
  m = 0
  for tri in tess; m += 1; end  # count number of triangles -- TODO
  t = zeros(Int64,m,3)
  k = 1
  for tri in tess
    t[k,:] = [tri._a._k, tri._b._k, tri._c._k]
    k = k + 1
  end
  (t, tess)
end

function triplot(tess)
  width = max_coord - min_coord
  x, y = getplotxy(delaunayedges(tess))
  plot(x,y,xlim=(min_coord,min_coord+width), ylim=(min_coord,min_coord+width))
end

function gradbasis(p,t)
  v = zeros(size(t,1),size(p,2),3)
  ∇φ = zeros(size(t,1),2,3)
  v[:,:,1] = p[t[:,3],:]-p[t[:,2],:]
  v[:,:,2] = p[t[:,1],:]-p[t[:,3],:]
  v[:,:,3] = p[t[:,2],:]-p[t[:,1],:]
  area = 0.5*(-v[:,1,3].*v[:,2,2] + v[:,2,3].*v[:,1,2])
  ∇φ[:,1,:], ∇φ[:,2,:] = -v[:,2,:]./(2*area), v[:,1,:]./(2*area)
  (∇φ, abs.(area))
end

function dotA2(x,A,y)
  Ay1 = A[:,1].*y[:,1] + A[:,2].*y[:,2]
  Ay2 = A[:,3].*y[:,1] + A[:,4].*y[:,2]
  x[:,1].*Ay1 + x[:,2].*Ay2
end

function assemble(p,t,pb,G)
  n = maximum(pb[:,2])
  dphi,area = gradbasis(p,t)
  D, M = spzeros(n,n), spzeros(n,n)
  Mij = area/12.*ones(size(dphi,1))
  for i = 1:3
    for j = i:3
      Dij = -area.*dotA2(dphi[:,:,i],G,dphi[:,:,j])
      I, J = pb[t[:,i],2], pb[t[:,j],2]
      if (j==i)
        D = D + sparse(I,J,Dij,n,n)
        M = M + sparse(I,J,2*Mij,n,n)
      else
        D = D + sparse([I;J],[J;I],[Dij; Dij],n,n)
        M = M + sparse([I;J],[J;I],[Mij; Mij],n,n)
      end
    end
  end
  (D,M)
end
