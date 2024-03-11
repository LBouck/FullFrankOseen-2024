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


max_iter = 1000000
minres_solve_tol = 1e-8
l2_err = []
hs = (1/sqrt(2))**np.arange(3,11)
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
    fname = "helene_stable_paper/tau="+str(tau)+"_h="+str(hmax)
    u0 = degree_n_defect(mesh, Vh, 1)
    dict = gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, k2 = .9, k3 = .85, frame_rate =  100, grad_flow_tol = grad_flow_tol,
                            minres_solve_tol = minres_solve_tol)

    u_sol = dict['Solution']
    exact = CoefficientFunction((x,y,z))
    exact = exact/Norm(exact)
    err2 = Norm(u_sol - exact)**2
    l2_err.append(sqrt(Integrate(err2,mesh)))
    print('L2 Error: '+str(l2_err[-1]))


np.savetxt("helene_stable_paper/space_refine_L2_errors.txt", np.array(l2_err))
