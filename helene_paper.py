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
from math import pi, sqrt
from gradient_flow import gradient_flow, nodal_interpolation_3
from defect_init import degree_n_defect



# sphere
geo = CSGeometry()
hmax = sqrt(2)/4
sphere = Sphere(Pnt(0,0,0), 1).bc('sphere')
mp = MeshingParameters(maxh=hmax)
geo.Add(sphere)
mesh = Mesh(geo.GenerateMesh(mp = mp))


max_iter = 1000000
minres_solve_tol = 1e-8
Vh = VectorH1(mesh, order = 1, dirichlet=("sphere"))
Qh = H1(mesh, order = 1, dirichlet=("sphere"))
Linf_err = []
L1_err = []
iterations = []
hs = (1/sqrt(2))**np.arange(3,10)
for hmax in hs:
    geo = CSGeometry()
    sphere = Sphere(Pnt(0,0,0), 1).bc('sphere')
    mp = MeshingParameters(maxh=hmax)
    geo.Add(sphere)
    mesh = Mesh(geo.GenerateMesh(mp = mp))
    Vh = VectorH1(mesh, order = 1, dirichlet=("sphere"))
    Qh = H1(mesh, order = 1, dirichlet=("sphere"))

    tau = hmax
    grad_flow_tol = (1e-3)/2*tau
    print("h = "+str(hmax)+" tau = "+str(tau))
    fname = "helene_paper/space_refine_tau="+str(tau)+"_h="+str(hmax)
    u0 = degree_n_defect(mesh, Vh, 1)
    dict = gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, k2 = .1, frame_rate =  100, grad_flow_tol = grad_flow_tol,
                            minres_solve_tol = minres_solve_tol)
    print('Iterations: '+str(dict['Iterations']))
    iterations.append(dict['Iterations'])
    print('Linf errors: '+str(dict['Linf errors'][-1]))
    Linf_err.append(dict['Linf errors'][-1])
    print('L1 errors: '+str(dict['L1 errors'][-1]))
    L1_err.append(dict['L1 errors'][-1])


np.savetxt("helene_paper/space_refine_L1_errors.txt", np.array(L1_err))
np.savetxt("helene_paper/space_refine_Linf_errors.txt", np.array(Linf_err))
np.savetxt("helene_paper/space_refine_iterations.txt", np.array(iterations))


# sphere
geo = CSGeometry()
hmax = 1/8
sphere = Sphere(Pnt(0,0,0), 1).bc('sphere')
mp = MeshingParameters(maxh=hmax)
geo.Add(sphere)
mesh = Mesh(geo.GenerateMesh(mp = mp))
Vh = VectorH1(mesh, order = 1, dirichlet=("sphere"))
Qh = H1(mesh, order = 1, dirichlet=("sphere"))
Linf_err = []
L1_err = []
iterations = []

for level in [0, 1, 2, 3, 4, 5]:
    tau = 4*hmax/(2**level)
    grad_flow_tol = (1e-3)/2*tau
    print("h = "+str(hmax)+" tau = "+str(tau))
    fname = "helene_paper/time_refine_tau="+str(tau)+"_h="+str(hmax)
    u0 = degree_n_defect(mesh, Vh, 1)
    dict = gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, k2 = .1, frame_rate =  100, grad_flow_tol = grad_flow_tol,
                            minres_solve_tol = minres_solve_tol)
    print('Iterations: '+str(dict['Iterations']))
    iterations.append(dict['Iterations'])
    print('Linf errors: '+str(dict['Linf errors'][-1]))
    Linf_err.append(dict['Linf errors'][-1])
    print('L1 errors: '+str(dict['L1 errors'][-1]))
    L1_err.append(dict['L1 errors'][-1])


np.savetxt("helene_paper/time_refine_L1_errors.txt", np.array(L1_err))
np.savetxt("helene_paper/time_refine_Linf_errors.txt", np.array(Linf_err))
np.savetxt("helene_paper/time_refine_iterations.txt", np.array(iterations))
