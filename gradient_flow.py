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




def nodal_interpolation_3(func,fes, bilin_form = None, zero_bc = False):

    gfu = GridFunction(fes)
    u,v = fes.TnT()
    vdual = v.Operator("dual")
    if bilin_form is None:
        a = BilinearForm(fes)
        a += u*vdual*dx(element_vb=BBBND)
        a.Assemble()
    else:
        a = bilin_form

    f = LinearForm(fes)
    f += func*vdual*dx(element_vb=BBBND)
    f.Assemble()
    if bilin_form is None:
        gfu.vec.data = a.mat.Inverse() * f.vec
    else:
        # gfu.Set(func, BND)
        # gfu.vec.data = a.mat.Inverse() * f.vec

        if zero_bc:
            with TaskManager():
                proj = Projector(fes.FreeDofs(), True)
                solvers.GMRes(A = a.mat, pre = proj, b = f.vec, x = gfu.vec,
                              maxsteps = 100000, printrates=False)
        else:
            proj = IdentityMatrix(fes.ndof, complex=False)
            solvers.GMRes(A = a.mat, pre = proj, b = f.vec, x = gfu.vec,
                    maxsteps = 100000, printrates=False)
        # gfu.Set(func, BND)
    return gfu


#@profile
def get_stiffness_inv(Vh, tau):
    '''returns stiffness matrix inverse operation on free dofs'''
    K = BilinearForm(Vh)
    K += (c0+1.0/tau)*InnerProduct(Grad(u),Grad(v))*dx
    K += c1*div(u)*div(v)*dx


    # c = Preconditioner(K, 'multigrid')
    K.Assemble()
    # inv = CGSolver(K.mat, c.mat, maxsteps=10000, printrates=False)
    # return inv
    Kinv = K.mat.Inverse(Vh.FreeDofs(), inverse="pardiso")
    return Kinv


# Kinv = get_stiffness_inv(Vh, tau)

# K = BilinearForm(Vh)
# K += (c0+1.0/tau)*InnerProduct(Grad(u),Grad(v))*dx
# K += c1*div(u)*div(v)*dx
# c = Preconditioner(K, 'local')
# K.Assemble()
# t2 = time.time()
# a.Assemble()
# f.Assemble()
#
# mp = BilinearForm(Qh)
# # mp += InnerProduct(grad(p),grad(q))*dx
# mp+= p*q*dx
#
# mp_inv = Preconditioner(mp, 'local')
# mp.Assemble()

# mp_inv = mp.mat.Inverse(Qh.FreeDofs(), inverse="pardiso")
# mp_inv = mp.mat
# M = BilinearForm(Qh)
# M += (c0+1.0/tau)*InnerProduct(grad(p),grad(q))*dx
# M.Assemble()
#
# Minv = M.mat.Inverse(Qh.FreeDofs(), inverse="pardiso")

# Identity matrix with free dofs; used as preconditioner
# proj = Projector(Qh.FreeDofs(), True)
# projV = Projector(Vh.FreeDofs(), True)
# # iteration
# energy_variation = 100000.0
# iter = 0

######
def magnetic_linear_form(u_old, Hfield, chi_a, Vh, u, v):
    magnetic = LinearForm(Vh)
    magnetic += chi_a*(Hfield*u_old)*(Hfield*v)*dx
    magnetic.Assemble()
    return magnetic


def splay_lin_form(u_old, Vh, u, v):
    splay = LinearForm(Vh)
    splay += div(u_old)*div(v)*dx
    splay.Assemble()
    return splay

def twist_lin_form(u_old, Vh, u, v):
    twist = LinearForm(Vh)

    # curl hand built for 3d
    # see https://ngsolve.org/media/kunena/attachments/889/curl_div.py
    curl_u = CoefficientFunction((Grad(u_old)[2,1]-Grad(u_old)[1,2],
                                  Grad(u_old)[0,2]-Grad(u_old)[2,0],
                                  Grad(u_old)[1,0]-Grad(u_old)[0,1]))
    curl_v = CoefficientFunction((Grad(v)[2,1]-Grad(v)[1,2],
                                  Grad(v)[0,2]-Grad(v)[2,0],
                                  Grad(v)[1,0]-Grad(v)[0,1]))

    twist += u_old*curl_u*( u_old*curl_v + v*curl_u)*dx
    twist.Assemble()
    return twist

def bend_lin_form(u_old, Vh, u, v):
    bend = LinearForm(Vh)

    # curl hand built for 3d
    # see https://ngsolve.org/media/kunena/attachments/889/curl_div.py
    curl_u = CoefficientFunction((Grad(u_old)[2,1]-Grad(u_old)[1,2],
                                  Grad(u_old)[0,2]-Grad(u_old)[2,0],
                                  Grad(u_old)[1,0]-Grad(u_old)[0,1]))
    curl_v = CoefficientFunction((Grad(v)[2,1]-Grad(v)[1,2],
                                  Grad(v)[0,2]-Grad(v)[2,0],
                                  Grad(v)[1,0]-Grad(v)[0,1]))

    bend += Cross(u_old,curl_u)*( Cross(u_old,curl_v) + Cross(v,curl_u))*dx
    bend.Assemble()
    return bend

def energy_components(u_old, Vh, u, v):

    splay = splay_lin_form(u_old, Vh, u, v)(u_old)
    twist = twist_lin_form(u_old, Vh, u, v)(u_old)/2
    bend = bend_lin_form(u_old, Vh, u, v)(u_old)/2
    return np.array([splay, twist, bend])


def bend_twist_lin_form(u_old, c2, c3, Vh, u, v):
    '''given vector valued u_old, this function returns a linear form
    to compute the contributions of the bending and twisting energies'''


    bend_twist = LinearForm(Vh)

    # curl hand built for 3d
    # see https://ngsolve.org/media/kunena/attachments/889/curl_div.py
    curl_u = CoefficientFunction((Grad(u_old)[2,1]-Grad(u_old)[1,2],
                                  Grad(u_old)[0,2]-Grad(u_old)[2,0],
                                  Grad(u_old)[1,0]-Grad(u_old)[0,1]))
    curl_v = CoefficientFunction((Grad(v)[2,1]-Grad(v)[1,2],
                                  Grad(v)[0,2]-Grad(v)[2,0],
                                  Grad(v)[1,0]-Grad(v)[0,1]))

    # # twisting contribution
    # bend_twist += c2*u_old*curl_u*( u_old*curl_v + v*curl_u)*dx
    #
    # # bending contribution
    # bend_twist += c3*Cross(u_old,curl_u)*( Cross(u_old,curl_v) + Cross(v,curl_u))*dx

    # more efficient implementation of bend and twist terms
    W_v = (Grad(v) - Grad(v).trans)/2
    e1 = CoefficientFunction((1,0,0))
    Umat = OuterProduct(e1, Cross(e1, u_old))
    e2 = CoefficientFunction((0,1,0))
    Umat += OuterProduct(e2, Cross(e2, u_old))
    e3 = CoefficientFunction((0,0,1))
    Umat += OuterProduct(e3, Cross(e3, u_old))
    # twisting contribution
    bend_twist += c2*u_old*curl_u*(InnerProduct(Umat,W_v) + v*curl_u)*dx
    # bending contribution
    bend_twist += c3*Cross(u_old,curl_u)*( -2*W_v*u_old + Cross(v,curl_u))*dx
    with TaskManager():
        bend_twist.Assemble()
    return bend_twist


# t2 = time.time()
# bend_twist = bend_twist_lin_form(u_old, c2, c3, Vh, u, v)
# print("Time to assemble nonlinear part: " + str(time.time()-t2))


def get_rhs(a, u_old, bend_twist, magnetic):
    '''returns upper half rhs of gradient flow'''
    return -a.mat*u_old.vec - bend_twist.vec + magnetic.vec


def constraint_matrix(u_old, Vh, Qh, u, q):
    '''returns constraint matrix transpose for the constraint u_old dot u_inc = 0'''
    massLumping = IntegrationRule(points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)],
                                  weights = [1/24, 1/24, 1/24, 1/24] )
    b = BilinearForm(trialspace=Vh, testspace=Qh)
    # intrule depricated, use SetIntegrationRule instead
    # b += SymbolicBFI(q * (u_old * u), intrule=massLumping)
    b += SymbolicBFI(q * (u_old * u)).SetIntegrationRule(ET.TET,massLumping)
    with TaskManager():
        b.Assemble()
    return b.mat

#@profile
def solve_saddle_system_minres(K, Kpre, Bmat, fvec, u_inc, mul_new, Qh, max_steps = 10000, Bpre= None, tol=1e-8):
    A = BlockMatrix( [ [K.mat, Bmat.T], [Bmat, None] ] )
    C = BlockMatrix( [ [Kpre, None], [None, Bpre] ] )
    rhs2 = LinearForm(Qh)
    rhs2.Assemble()
    rhs_block = BlockVector ( [fvec, rhs2.vec] )
    sol_block = BlockVector( [u_inc.vec, mul_new.vec] )
    with TaskManager():
        solvers.MinRes(mat = A, pre = C, rhs = rhs_block, sol = sol_block,
                       maxsteps = max_steps, printrates=False, initialize=True, tol=tol)


#######

# bend and twist terms linear form
# bend_twist = bend_twist_lin_form(u_old, c2, c3, Vh, u, v)
# magnetic = magnetic_linear_form(u_old, Hfield, chi_a, Vh, u, v)
#
# # energy
# energy = a.Energy(u_old.vec) + bend_twist(u_old)/4 - magnetic(u_old)/2
#
# agrad = BilinearForm(Vh)
# agrad += c0*InnerProduct(Grad(u),Grad(v))*dx
# agrad.Assemble()
# adiv = BilinearForm(Vh)
# adiv += c1*(div(u)*div(v))*dx
# adiv.Assemble()
# print ("energy = ", energy)
# print("bend_twist = "+str(bend_twist(u_old, Vh, u, v)/4))
# print("grad = "+str(agrad.Energy(u_old.vec)))
# print("div = "+str(adiv.Energy(u_old.vec)))
# print("div = "+str(splay_lin_form(u_old, Vh, u, v)(u_old)))
# np.savetxt(fname+"_energy_components_init.txt", energy_components(u_old))
# energies = []
# energies.append(energy)
# Linftyh_errors = []
# IhD = nodal_interpolation_3(Norm(Norm(u_old)*Norm(u_old) - 1.0), Qh, bilin_form = nodal_bilinQ, zero_bc = True)
# Linftyh_errors.append(max(IhD.vec))





def gradient_flow(mesh, Vh, Qh, hmax, tau, u0, fname, frame_rate = 10, k1=1, k2=1, k3=1, chi_a = 0,
                  Hfield = CoefficientFunction((0,0,0)),
                  grad_flow_tol = 1e-4, minres_solve_tol = 1e-8, max_iter = 10000, minres_max_iter = 10000,
                  printprogress = True, printrates = False, Kpre = 'local', Bpre = 'local',
                  minres_tol_inexact = 0):


    t0 = time.time()
    # test and trial functions
    u,v = Vh.TnT()
    p,q = Qh.TnT()

    # modified Frank's constants
    c0 = min(k1,k2,k3)
    c1 = k1 - c0
    c2 = k2 - c0
    c3 = k3 - c0



    params = np.array([hmax, tau, grad_flow_tol, frame_rate, k1, k2, k3, chi_a])

    print("saving to: "+fname)
    np.savetxt(fname+"_params.txt", params)

    print("degrees of freedom")
    print(Vh.ndof)

    # blocks and rhs
    a = BilinearForm(Vh)
    b = BilinearForm(trialspace=Vh, testspace=Qh)
    f = LinearForm(Vh)

    # bilinear forms and rhs
    a += c0*InnerProduct(Grad(u),Grad(v))*dx # a:Frank's energy
    a += c1*div(u)*div(v)*dx

    K = BilinearForm(Vh)
    K += (c0+1.0/tau)*InnerProduct(Grad(u),Grad(v))*dx
    K += c1*div(u)*div(v)*dx
    c = Preconditioner(K, Kpre)
    K.Assemble()
    t2 = time.time()
    a.Assemble()
    f.Assemble()

    mp = BilinearForm(Qh)
    # mp += InnerProduct(grad(p),grad(q))*dx
    mp+= p*q*dx

    mp_inv = Preconditioner(mp, Bpre)
    mp.Assemble()



    vdual = v.Operator("dual")
    nodal_bilinV = BilinearForm(Vh)
    nodal_bilinV += u*vdual*dx(element_vb=BBBND)
    nodal_bilinV.Assemble()

    qdual = q.Operator("dual")
    nodal_bilinQ = BilinearForm(Qh)
    nodal_bilinQ += p*qdual*dx(element_vb=BBBND)
    nodal_bilinQ.Assemble()


    u_old = GridFunction(Vh)
    u_old.vec.data = u0.vec

    vtk = VTKOutput(ma=mesh,
                    coefs=[u_old],
                    names = ["sol"],
                    filename=fname+"",
                    subdivision=0)
    vtk.Do()


    IhD = nodal_interpolation_3(Norm(Norm(u_old)*Norm(u_old) - 1.0), Qh, bilin_form = nodal_bilinQ, zero_bc = True)
    l1herror = Integrate((IhD),mesh)
    print("L1h Error of director length: {0}".format(l1herror))
    Linftyh_error = max(IhD.vec)
    print("Linftyh Error of director length: {0}".format(Linftyh_error))
    Linftyh_errors = []
    L1h_errors =  []
    Linftyh_errors.append(Linftyh_error)

    l1herror = Integrate((IhD),mesh)
    print("L1h Error of director length: {0}".format(l1herror))
    L1h_errors.append(l1herror)
    # solution vector
    u_new = GridFunction(Vh) #new solution
    u_inc = GridFunction(Vh) #solution increment
    mul_new = GridFunction(Qh) #multiplier

    energy_variation = 10000

    # bend and twist terms linear form
    bend_twist = bend_twist_lin_form(u_old, c2, c3, Vh, u, v)
    magnetic = magnetic_linear_form(u_old, Hfield, chi_a, Vh, u, v)
    # energy
    np.savetxt(fname+"_energy_components_init.txt", energy_components(u_old, Vh, u, v))
    energy = a.Energy(u_old.vec) + bend_twist(u_old)/4 - magnetic(u_old)/2
    energies = []
    energies.append(energy)
    iter = 0
    while abs(energy_variation) > grad_flow_tol:
        energy_variation = -energy

        Bmat = constraint_matrix(u_old, Vh, Qh, u, q)
        f.vec.data = get_rhs(a, u_old, bend_twist, magnetic)

        T0 = time.time()
        # print(mul_new.vec.data)
        # u_inc.vec.data = solve_saddle_system(Kinv, Bmat, f.vec.data, mul_new)
        # solv_tol = tau*min(max(-energy_variation, tol), 1e-3)
        # print("solve tol: "+str(solv_tol))
        solve_saddle_system_minres(K, c, Bmat, f.vec, u_inc, mul_new, Qh,
                                   max_steps = minres_max_iter, Bpre = mp_inv,
                                   tol=minres_solve_tol)
        # u_inc.vec.data = solve_saddle_system_CG_Schur(K.mat, c, Bmat, f.vec, tol=1e-14)
        print("Solve time: "+str(time.time()-T0))
        u_new.vec.data = u_old.vec + u_inc.vec
        u_old.vec.data = u_new.vec
        iter += 1

        # create new bend and twist linear form
        bend_twist = bend_twist_lin_form(u_old, c2, c3, Vh, u, v)
        magnetic = magnetic_linear_form(u_old, Hfield, chi_a, Vh, u, v)
        # Compute energy and energy variation
        energy = a.Energy(u_old.vec) + bend_twist(u_old)/4 - magnetic(u_old)/2
        print ("energy = ", energy)
        energy_variation += energy

        energies.append(energy)
        IhD = nodal_interpolation_3(Norm(Norm(u_old)*Norm(u_old) - 1.0), Qh, bilin_form = nodal_bilinQ,zero_bc = True)
        Linftyh_errors.append(max(IhD.vec))
        print ("energy variation = ", -energy_variation)
        print("Progress: "+str(np.log(-energy_variation)/np.log(grad_flow_tol)))
        print("Linftyh Error of director length: {0}".format(Linftyh_errors[-1]))
        l1herror = Integrate((IhD),mesh)
        print("L1h Error of director length: {0}".format(l1herror))
        L1h_errors.append(l1herror)
        print("count: "+ str(iter))

        if minres_tol_inexact == 0:
            minres_solve_tol = minres_solve_tol
        else:
            minres_solve_tol = minres_tol_inexact*sqrt(Integrate(Norm(u_inc)**2,mesh))
            print('minres_solve_tol = '+str(minres_solve_tol))

        if iter%frame_rate == 0:
            vtk.Do()

        if iter > max_iter:
            break
        if energy_variation > 0:
            print("\n\n\n***FAILED TO DECREASE ENERGY. PICK SMALLER TAU***\n\n\n")
            break

    vtk.Do()
    IhD = nodal_interpolation_3(Norm(Norm(u_old)*Norm(u_old) - 1.0), Qh, bilin_form = nodal_bilinQ, zero_bc = True)
    l1herror = Integrate(Norm(IhD),mesh)
    print("L1h Error of director length: {0}".format(l1herror))
    # Linftyh_errors.append(max(nodal_interpolation_3(Norm(IhD),Qh).vec))
    print("Linftyh Error of director length: {0}".format(Linftyh_errors[-1]))
    L1h_errors.append(l1herror)
    print ("energy = ", energy)


    t1 = time.time()
    print("total time elapsed: "+str(t1-t0))

    np.savetxt(fname+"_energies.txt", np.array(energies))
    np.savetxt(fname+"_Linf_errors.txt", np.array(Linftyh_errors))
    np.savetxt(fname+"_energy_components_final.txt", energy_components(u_old, Vh, u, v))
    np.savetxt(fname+"_L1_errors.txt", np.array(L1h_errors))
    np.savetxt(fname+"_cpu_time.txt", np.array([t1-t0]))

    return {'Solution':u_old, 'Energies': np.array(energies),
            'Linf errors': np.array(Linftyh_errors), 'L1 errors': np.array(L1h_errors),
            'Iterations': iter, 'Time': t1-t0}
