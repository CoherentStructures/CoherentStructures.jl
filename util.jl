


#Based on JuAFEM's WriteVTK.vtk_point_data 
function fixU(dh::DofHandler, u::Vector)
    res = fill(0.0,getnnodes(dh.grid))
    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        counter = 1
        offset = JuAFEM.field_offset(dh, dh.field_names[1])
       for node in getnodes(cell)
               res[node] = u[_celldofs[counter + offset]]
               counter += 1
          end
      end
      return res
  end
