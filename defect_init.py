import cmath
import numpy as np
from numpy.linalg import norm
from netgen.csg import *
from netgen.meshing import MeshingParameters
import netgen.gui
from ngsolve import *
# sterographic projection
def stereo(x):
    return x[0:2]/(1-x[2])
# inverse of stereographic projection
def stereo_inv(x):
    y = np.zeros(3)
    normx2 = norm(x)**2
    y[0:2] = 2*x/(1+normx2)
    y[2] = (normx2-1)/(normx2+1)
    return y
# take a vector in 2d and apply power to the n as if it were a complex number
def complex_pow(x, n):
    z = complex(x[0],x[1])
    zn = z**n
    return np.array([zn.real,zn.imag])

def degree_n_defect_point(x, n):
    '''Given point x. Returns a vector value that comes from a degree n
       defect'''
    x = x/norm(x)
    # if the vector is pointing straight up or down, do nothing
    if norm(x - np.array([0,0,1])) == 0:
        return x
    if norm(x + np.array([0,0,1])) == 0:
        return x
    # otherwise
    return stereo_inv(complex_pow(stereo(x), n))


def degree_n_defect(mesh, Vh, n):
    '''Given mesh and space. Returns a function that is the lagrange interpolant
       of a degree n defect'''
    nodal_values = []
    for node in mesh.vertices:
        val = degree_n_defect_point(np.asarray(node.point), n)
        nodal_values.append(val/norm(val))

    nodal_values = np.asarray(nodal_values).flatten('F')

    u_init = GridFunction(Vh)
    u_init.vec.FV().NumPy()[:] = nodal_values

    return u_init
