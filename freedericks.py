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

# freedericks geometry
geo = CSGeometry()
left  = Plane (Pnt(0,0,0), Vec(-1,0,0) )
right = Plane (Pnt(1,1,1), Vec( 1,0,0) )
front = Plane (Pnt(0,0,0), Vec(0,-1,0) )
back  = Plane (Pnt(1,1,1), Vec(0, 1,0) )
bot   = Plane (Pnt(0,0,0), Vec(0,0,-1) ).bc('bottom')
top   = Plane (Pnt(1,1,.5), Vec(0,0, 1) ).bc('top')
tall_box = (left * right * front * back)
closed_box = tall_box * bot * top
geo.Add (closed_box)
hmax = 1/32
ngmesh = geo.GenerateMesh(maxh=hmax)
mesh = Mesh(ngmesh)
Draw(mesh)



Vh = VectorH1(mesh, order = 1, dirichlet=("top|bottom"))
Qh = H1(mesh, order = 1, dirichlet=("top|bottom"))

# data
Hfield = CoefficientFunction((0,0,1))*9.5

pfunc = 128*CoefficientFunction((x*(1-x)*y*(1-y))**2*z*(.5-z))*Hfield/Norm(Hfield)
bcfunc1 = CoefficientFunction((1,0,0))
u_old_exact = bcfunc1 + pfunc
u_old_exact = u_old_exact/Norm(u_old_exact)

u,v = Vh.TnT()
vdual = v.Operator("dual")
nodal_bilinV = BilinearForm(Vh)
nodal_bilinV += u*vdual*dx(element_vb=BBBND)
nodal_bilinV.Assemble()

u0 = nodal_interpolation_3(u_old_exact,Vh, bilin_form = nodal_bilinV)


# paa at 122 deg C
k1 = 2.3
k2 = 1.5
k3 = 4.8
chi_a = 1.21
C = 1
frame_rate = 5
tau = C*hmax
max_iter = 1000000
grad_flow_tol = (1e-4)/2*tau
print("h = "+str(hmax)+" tau = "+str(tau))

fname = "freedericks_paper/splay"


dict = gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, frame_rate = 5, k1=k1,
                    k2=k2, k3=k3, chi_a = chi_a, Hfield = Hfield, grad_flow_tol = grad_flow_tol)
