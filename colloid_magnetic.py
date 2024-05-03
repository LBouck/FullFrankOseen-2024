import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import matplotlib.pyplot as plt

from netgen.csg import *
from netgen.meshing import MeshingParameters
import netgen.gui
from ngsolve import *
from ngsolve.la import EigenValues_Preconditioner
import time
from math import pi
from gradient_flow import gradient_flow, nodal_interpolation_3




# colloid
geo = CSGeometry()
hmax = 1/8
brick = OrthoBrick( Pnt(-2, -2, -2), Pnt(2,2,2)).bc('outside')
sphere = Sphere(Pnt(0,0,0), .75).bc('sphere')#.maxh(hmax/2)
mp = MeshingParameters(maxh=hmax)
geo.Add(brick-sphere)
mesh = Mesh(geo.GenerateMesh(mp = mp))
Draw(mesh)


# parameters

t0 = time.time()
order = 1
C = 1/4
tau = C*hmax
max_iter = 1000000
grad_flow_tol = (1e-4)*tau
minres_solve_tol = 1e-7
frame_rate = 20
print("h = "+str(hmax)+" tau = "+str(tau))
fname = "colloid_magnetic/tau="+str(tau)+"_h="+str(hmax)

Vh = VectorH1(mesh, order = order, dirichlet=("outside|sphere"))
Qh = H1(mesh, order = order, dirichlet=("outside|sphere"))

# for colloid experiment
r = sqrt(x*x+y*y+z*z)
bcfunc_box = CoefficientFunction((r-.75)*(0,0,1))
bcfunc_sphere = CoefficientFunction((x,y,z))
bcfunc_sphere = bcfunc_sphere*CoefficientFunction((2-x)*(2+x)*(2-y)*(2+y)*(2-z)*(2+z))
u_old_exact = bcfunc_sphere + 100*bcfunc_box
u_old_exact = u_old_exact/Norm(u_old_exact)
u,v = Vh.TnT()
vdual = v.Operator("dual")
nodal_bilinV = BilinearForm(Vh)
nodal_bilinV += u*vdual*dx(element_vb=BBBND)
nodal_bilinV.Assemble()

u0 = nodal_interpolation_3(u_old_exact,Vh, bilin_form = nodal_bilinV)

for H in [0,1,2,4]:
    Hfield = CoefficientFunction((0,H,0))
    fname = "colloid_magnetic/H="+str(H)+"_tau="+str(tau)+"_h="+str(hmax)
    dict1 = gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, frame_rate =  frame_rate,
                        chi_a = 1, Hfield = Hfield, grad_flow_tol = grad_flow_tol,
                        minres_solve_tol = minres_solve_tol)
