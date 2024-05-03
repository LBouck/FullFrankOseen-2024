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
from defect_init import degree_n_defect



# colloid
geo = CSGeometry()
hmax = 1/16
sphere = Sphere(Pnt(0,0,0), 1).bc('sphere')
mp = MeshingParameters(maxh=hmax)
geo.Add(sphere)
mesh = Mesh(geo.GenerateMesh(mp = mp))
Draw(mesh)


# parameters

t0 = time.time()
order = 1
C = 1
tau = C*hmax
max_iter = 1000000
grad_flow_tol = (1e-4)*tau
minres_solve_tol = 1e-8
print("h = "+str(hmax)+" tau = "+str(tau))
Vh = VectorH1(mesh, order = order, dirichlet=("outside|sphere"))
Qh = H1(mesh, order = order, dirichlet=("outside|sphere"))

# initial condition
u0 = degree_n_defect(mesh, Vh, 2)

fname = "degree2_paper/one_const_tau="+str(tau)+"_h="+str(hmax)
dict_one_const = gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, frame_rate =  10, grad_flow_tol = grad_flow_tol,
                        minres_solve_tol = minres_solve_tol)


k1=1
k2 = .75
for k3 in [1,3,5]:
    fname = "degree2_paper/k3="+str(k3)+"_tau="+str(tau)+"_h="+str(hmax)
    dict = gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, k1=k1, k2=k2, k3=k3,
                                frame_rate =  10, grad_flow_tol = grad_flow_tol,
                                minres_solve_tol = minres_solve_tol)
