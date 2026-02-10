# General.
import numpy as np
import csv, glob, os, re, io
import timeit as tt
import struct
import scipy.sparse as sp

from enum import IntEnum
import multiprocessing

# Netgen.
import ngsolve as ng
from netgen.meshing import Element2D, Element3D, MeshPoint, Mesh
from netgen.csg import Pnt

# Carp.
from carputils.carpio import igb
from carputils.carpio import txt


def swapPositions(list, pos1, pos2):
    """ Swap positions. """
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


class CarpElemType(IntEnum):
    """ Enumeration class. """
    Ln = 1
    Qd = 2
    Tr = 3
    Tt = 4
    Hx = 5
    Py = 6
    Pr = 7
    Oc = 8


def get_ndofs(elemtype):
    """ Return number of DoFs """
    if elemtype == CarpElemType.Ln:
        return 2
    elif elemtype == CarpElemType.Tr:
        return 3
    elif elemtype == CarpElemType.Qd:
        return 4
    elif elemtype == CarpElemType.Tt:
        return 4
    elif elemtype == CarpElemType.Hx:
        return 8
    else:
        print("ERROR elemtype ", elemtype, " not recognized!")
        return 0


def _h5_binary(ds):
    return io.BytesIO(ds[()].tobytes())


def _h5_text(ds):
    return io.StringIO(ds[()].tobytes().decode("utf8"))


def discover_mesh(mesh_grp):

    keys = list(mesh_grp.keys())

    meshname = os.path.commonprefix(keys).rstrip("_ .")

    surfs = sorted(k for k in keys
                   if k.startswith(meshname) and k.endswith(".surf"))

    return meshname, surfs


def read_carp_mesh(mesh_grp):

    xyz, e2n, tags = None, None, None

    meshname, surflist = discover_mesh(mesh_grp)

    if meshname + '.belem' in mesh_grp.keys():
        xyz, e2n, tags = read_carp_bin_mesh(mesh_grp, meshname)
    else:
        raise RuntimeError(f"binary mesh not found with name: {meshname}")

    surfdata = [mesh_grp[s] for s in surflist]

    return xyz, e2n, tags, surfdata


def read_carp_bin_mesh(mesh_grp, meshname):

    # ---------- points ----------
    f = _h5_binary(mesh_grp[meshname + ".bpts"])
    numpts, endianness, _ = map(
        int,
        f.read(1024).decode("ascii").strip("\x00").split())
    bo = "<" if endianness == 0 else ">"
    xyz = np.frombuffer(f.read(), dtype=f"{bo}f4").reshape(numpts, 3) * 1e-3

    # ---------- elements ----------
    f = _h5_binary(mesh_grp[meshname + ".belem"])
    numele, endianness, _ = map(
        int,
        f.read(1024).decode("ascii").strip("\x00").split())
    bo = "<" if endianness == 0 else ">"
    tmp = np.frombuffer(f.read(), dtype=f"{bo}i4").reshape(numele, -1)

    e2n = tmp[:, 1:5]
    tags = tmp[:, -1]

    return xyz, e2n, tags


def parse_carp_surface(ds):

    surf_e2n = []
    surf_ndof = []

    f = _h5_text(ds)
    reader = csv.reader(f)

    numele = int(next(reader)[0])

    for row in reader:
        tmp = row[0].split()
        elemtype = CarpElemType[tmp[0]]
        nd = get_ndofs(elemtype)

        nodes = [int(tmp[k + 1]) for k in range(nd)]

        if elemtype == CarpElemType.Tr:
            swapPositions(nodes, 0, 1)

        surf_ndof.append(nd)
        surf_e2n.append(nodes)

    return surf_e2n, surf_ndof


def carp_mesh_to_ngmesh(xyz, e2n, etags, surfdata, surfnames, scaling=1.0):

    xyz *= scaling
    mesh = Mesh()

    tags = sorted(set(etags))
    tagmap = {t: i + 1 for i, t in enumerate(tags)}

    mesh.dim = 3

    regions = {t: mesh.AddRegion(f"tag_region_{t}", dim=3) for t in tags}

    pnums = [mesh.Add(MeshPoint(Pnt(x))) for x in xyz]

    for dofs, tag in zip(e2n, etags):
        mesh.Add(Element3D(tagmap[tag], [pnums[k] for k in dofs]))

    for ds, name in zip(surfdata, surfnames):

        surfID = mesh.AddRegion(name, dim=2)

        surf_e2n, surf_ndof = parse_carp_surface(ds)

        for nodes, nd in zip(surf_e2n, surf_ndof):
            mesh.Add(Element2D(surfID, [pnums[k] for k in nodes[:nd]]))

    return ng.Mesh(mesh)


def get_rieszReps_hdf5(mesh_data_hdf5_file, riesz_rep_hdf5_file):
    """
    Load Riesz representers and the space matrix.

    :param hdf5_file: the file to load stuff from
    
    :return: the Riesz representers and the 1d space matrix.
    :rtype: numpy.ndarray, numpy.ndarray.
    """

    # not break existing code
    import h5py
    start = tt.default_timer()

    stiffness = None
    ndofs1d = None

    with h5py.File(riesz_rep_hdf5_file,
                   'r') as sol_h5, h5py.File(mesh_data_hdf5_file,
                                             'r') as mesh_h5:
        # mesh group
        if "mesh" in mesh_h5:
            mgrp = mesh_h5['mesh']
            xyz = mgrp['xyz'][:]
            e2n = mgrp['e2n'][:]
            tags = mgrp['tags'][:]
        # mtx group
        if "stiffness_matrix" in mesh_h5:
            matrix_group = mesh_h5["stiffness_matrix"]
            data = matrix_group["data"][:]
            indices = matrix_group["indices"][:]
            indptr = matrix_group["indptr"][:]
            shape = matrix_group.attrs["shape"]
            ndofs1d = shape[0]
            # Reconstruct sparse matrix
            stiffness = sp.csr_matrix((data, indices, indptr),
                                      shape=shape)  # no permutation needed
        # load the rieszReps
        if "riesz_reps" in sol_h5:
            riesz_sol = sol_h5['riesz_reps']
            all_keys = list(riesz_sol.keys())
            total_solutions = len(all_keys)
            rieszReps = np.empty((total_solutions, ndofs1d))
            for i, k in enumerate(all_keys):
                rieszReps[i, :] = riesz_sol[k][:]

    end = tt.default_timer()
    print('load space matrix and riesz elements:', end - start, 's')
    return rieszReps, stiffness


def save_sparse_csr(filename, array):
    np.savez(filename,
             data=array.data,
             indices=array.indices,
             indptr=array.indptr,
             shape=array.shape)


def load_sparse_csr(file_h5):
    return sp.csr_matrix(
        (file_h5['data'], file_h5['indices'], file_h5['indptr']),
        shape=file_h5['shape'])
