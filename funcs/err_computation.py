import numpy as np
import ngsolve as ng


def compute_errors(u_rec, u_true, fes, mesh):
    diff = ng.GridFunction(fes)
    u = ng.GridFunction(fes)
    diff.vec.FV().NumPy()[:] = u_true - u_rec
    u.vec.FV().NumPy()[:] = u_true
    numL2 = ng.Integrate(ng.InnerProduct(diff, diff), mesh)
    denL2 = ng.Integrate(ng.InnerProduct(u, u), mesh)
    errL2 = np.sqrt(numL2 / denL2)
    numH10 = ng.Integrate(ng.InnerProduct(diff.Deriv(), diff.Deriv()), mesh)
    denH10 = ng.Integrate(ng.InnerProduct(u.Deriv(), u.Deriv()), mesh)
    errH10 = np.sqrt(numH10 / denH10)
    errH1 = np.sqrt((numL2 + numH10) / (denL2 + denH10))

    p = 25.0
    errLinfapprox = ng.Integrate(ng.sqrt(ng.InnerProduct(diff, diff))**p,
                                 mesh)**(1. / p)
    errLinfden = ng.Integrate(ng.sqrt(ng.InnerProduct(u, u))**p, mesh)**(1. / p)
    errLinfty = errLinfapprox / errLinfden

    return errL2, errH1, errH10, errLinfty
