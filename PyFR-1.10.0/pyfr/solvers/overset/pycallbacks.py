# -*- coding: utf-8 -*-
from mpi4py import MPI
import tioga as tg
from convert import *
import numpy as np
from collections import defaultdict, OrderedDict

def _get_inter_objs(interside, getter, elemap):
    emap = {type: getattr(ele, getter) for type, ele in elemap.items()}
    return [emap[type](eidx, fidx) for type, eidx, fidx, flags in interside]

def face_vidx_incell():
    # IJK indexing to gmsh format, face pointing inward
    facemaps = defaultdict()
       
    map_tri_2  = {}
    map_tri_3  = {}
    map_tri_4  = {}

    map_quad_2 = {0:[0,1],1:[1,2],2:(2,3),3:[3,0]}
    map_quad_3 = {0:[0,1,2],1:[2,5,8],2:[8,7,6],3:[6,3,0]}
    map_quad_4 = {0:[0,1,2,3],1:[3,7,11,15],2:[15,14,13,12],3:[12,8,4,0]}

    map_hex_2 = {}
    map_hex_3 = {0:[0,2,8,6,1,5,7,3,4],
                 1:[2,0,18,20,1,9,19,11,10],
                 2:[8,2,20,26,5,11,23,17,14],
                 3:[6,8,26,24,7,17,25,15,16],
                 4: [0,6,24,18,3,15,21,9,12],
                 5:[20,18,24,26,19,21,25,23,22]}
    map_hex_4 = {}
    
    map_pri_2 = {}
    map_pri_3 = {}
    map_pri_4 = {}

    map_tet_2 = {}
    map_tet_3 = {}
    map_tet_4 = {}

    facemaps['tri_2'] = map_tri_2
    facemaps['tri_3'] = map_tri_3
    facemaps['tri_4'] = map_tri_4

    facemaps['quad_2'] = map_quad_2
    facemaps['quad_3'] = map_quad_3
    facemaps['quad_4'] = map_quad_4
        
    facemaps['hex_2'] = map_hex_2
    facemaps['hex_3'] = map_hex_3
    facemaps['hex_4'] = map_hex_4

    facemaps['pri_2'] = map_pri_2
    facemaps['pri_3'] = map_pri_3
    facemaps['pri_4'] = map_pri_4

    facemaps['tet_2'] = map_tet_2
    facemaps['tet_3'] = map_tet_2
    facemaps['tet_4'] = map_tet_2

    return facemaps


def gmsh_to_structured_quad(nNodes):
    '''
    function to convert from gmsh to stuctured notation
    '''
    gmsh_to_ijk = [0]*nNodes

    if nNodes != 8:
        nNodes1D = int(np.sqrt(nNodes))
        assert nNodes1D*nNodes1D != nNodes, 'nNodes must be a square number'

        nLevels = nNodes1D // 2
        
        node = 0
        for i in range(nLevels):
            i2 = nNodes1D - 1 -i
            gmsh_to_ijk[node + 0] = i + nNodes1D*i
            gmsh_to_ijk[node + 1] = i2 + nNodes1D * i
            gmsh_to_ijk[node + 2] = i2 + nNodes1D * i2
            gmsh_to_ijk[node + 3] = i  + nNodes1D * i2
            
            node += 4
            nEdgeNodes = nNodes1D - 2 * (i + 1);
            for j in range(nEdgeNodes):
                gmsh_to_ijk[node + 0*nEdgeNodes + j] = i+1+j  + nNodes1D * i
                gmsh_to_ijk[node + 1*nEdgeNodes + j] = i2     + nNodes1D * (i+1+j)
                gmsh_to_ijk[node + 2*nEdgeNodes + j] = i2-1-j + nNodes1D * i2
                gmsh_to_ijk[node + 3*nEdgeNodes + j] = i      + nNodes1D * (i2-1-j)

            node += 4 * nEdgeNodes;
        if nNodes1D % 2 != 0:
            gmsh_to_ijk[nNodes - 1] = nNodes1D//2 + nNodes1D * (nNodes1D//2)
    else:
        gmsh_to_ijk[0] = 0
        gmsh_to_ijk[1] = 2
        gmsh_to_ijk[2] = 7
        gmsh_to_ijk[3] = 5
        gmsh_to_ijk[4] = 1
        gmsh_to_ijk[5] = 3
        gmsh_to_ijk[6] = 4
        gmsh_to_ijk[7] = 6

    return gmsh_to_ijk

def structured_hex_to_gmsh(order):
    gmsh_to_ijk = gmsh_to_structured_hex(order)
    ijk_to_gmsh = [gmsh_to_ijk.index(i) for i in range(order*order*order)]
    return ijk_to_gmsh

def gmsh_to_structured_hex(order):
    nNodes = order*order*order
    gmsh_to_ijk = [0]*order*order*order
    nSide = order
    
    nLevels = nSide // 2
    isOdd = nSide % 2

    nPts = 0
    for i in range(nLevels):
        #Corners
        i2 = (nSide-1) - i
        gmsh_to_ijk[nPts+0] = i  + nSide * (i  + nSide * i)
        gmsh_to_ijk[nPts+1] = i2 + nSide * (i  + nSide * i)
        gmsh_to_ijk[nPts+2] = i2 + nSide * (i2 + nSide * i)
        gmsh_to_ijk[nPts+3] = i  + nSide * (i2 + nSide * i)
        gmsh_to_ijk[nPts+4] = i  + nSide * (i  + nSide * i2)
        gmsh_to_ijk[nPts+5] = i2 + nSide * (i  + nSide * i2)
        gmsh_to_ijk[nPts+6] = i2 + nSide * (i2 + nSide * i2)
        gmsh_to_ijk[nPts+7] = i  + nSide * (i2 + nSide * i2)
        nPts += 8

        # Edges
        nSide2 = nSide - 2 * (i+1)
        for j in range(nSide2):
            #Edges around 'bottom'
            gmsh_to_ijk[nPts+0*nSide2+j] = i+1+j  + nSide * (i     + nSide * i)
            gmsh_to_ijk[nPts+3*nSide2+j] = i2     + nSide * (i+1+j + nSide * i)
            gmsh_to_ijk[nPts+5*nSide2+j] = i2-1-j + nSide * (i2    + nSide * i)
            gmsh_to_ijk[nPts+1*nSide2+j] = i      + nSide * (i+1+j + nSide * i)

            # 'Vertical' edges
            gmsh_to_ijk[nPts+2*nSide2+j] = i  + nSide * (i  + nSide * (i+1+j))
            gmsh_to_ijk[nPts+4*nSide2+j] = i2 + nSide * (i  + nSide * (i+1+j))
            gmsh_to_ijk[nPts+6*nSide2+j] = i2 + nSide * (i2 + nSide * (i+1+j))
            gmsh_to_ijk[nPts+7*nSide2+j] = i  + nSide * (i2 + nSide * (i+1+j))

            #Edges around 'top'
            gmsh_to_ijk[nPts+ 8*nSide2+j] = i+1+j  + nSide * (i     + nSide * i2)
            gmsh_to_ijk[nPts+10*nSide2+j] = i2     + nSide * (i+1+j + nSide * i2)
            gmsh_to_ijk[nPts+11*nSide2+j] = i2-1-j + nSide * (i2    + nSide * i2)
            gmsh_to_ijk[nPts+ 9*nSide2+j] = i      + nSide * (i+1+j + nSide * i2)

        nPts += 12*nSide2;

        nLevels2 = nSide2 // 2
        isOdd2 = nSide2 % 2
        
        for j0 in range(nLevels2):
            # Corners
            j = j0 + i + 1;
            j2 = i + 1 + (nSide2-1) - j0;
            gmsh_to_ijk[nPts+0] = j  + nSide * (j  + nSide * i)
            gmsh_to_ijk[nPts+1] = j  + nSide * (j2 + nSide * i)
            gmsh_to_ijk[nPts+2] = j2 + nSide * (j2 + nSide * i)
            gmsh_to_ijk[nPts+3] = j2 + nSide * (j  + nSide * i)
            nPts += 4

            # Edges: Bottom, right, top, left
            nSide3 = nSide2 - 2 * (j0+1)
            for k in range(nSide3):
                gmsh_to_ijk[nPts+0*nSide3+k] = j      + nSide * (j+1+k  + nSide * i)
                gmsh_to_ijk[nPts+1*nSide3+k] = j+1+k  + nSide * (j2     + nSide * i)
                gmsh_to_ijk[nPts+2*nSide3+k] = j2     + nSide * (j2-1-k + nSide * i)
                gmsh_to_ijk[nPts+3*nSide3+k] = j2-1-k + nSide * (j      + nSide * i)
            nPts += 4*nSide3;

        # Center node for even-ordered Lagrange quads (odd value of nSide)
        if isOdd2:
            gmsh_to_ijk[nPts] = nSide // 2 +  nSide*(nSide // 2) +  nSide*nSide*i
            nPts += 1;

        # --- Front face ---
        for j0 in range(nLevels2):
            # Corners
            j = j0 + i + 1
            j2 = i + 1 + (nSide2-1) - j0
            gmsh_to_ijk[nPts+0] = j  + nSide * (i + nSide * j)
            gmsh_to_ijk[nPts+1] = j2 + nSide * (i + nSide * j)
            gmsh_to_ijk[nPts+2] = j2 + nSide * (i + nSide * j2)
            gmsh_to_ijk[nPts+3] = j  + nSide * (i + nSide * j2)
            nPts += 4

            # Edges: Bottom, right, top, left
            nSide3 = nSide2 - 2 * (j0+1)
            for k in range(nSide3):
                gmsh_to_ijk[nPts+0*nSide3+k] = j+1+k  + nSide * (i + nSide * j)
                gmsh_to_ijk[nPts+1*nSide3+k] = j2     + nSide * (i + nSide * (j+1+k))
                gmsh_to_ijk[nPts+2*nSide3+k] = j2-1-k + nSide * (i + nSide * j2)
                gmsh_to_ijk[nPts+3*nSide3+k] = j      + nSide * (i + nSide * (j2-1-k))
            nPts += 4*nSide3

        # Center node for even-ordered Lagrange quads (odd value of nSide)
        if isOdd2:
            gmsh_to_ijk[nPts] = nSide // 2 + nSide*(i + nSide*(nSide // 2))
            nPts += 1

        # --- Left face ---
        for j0 in range(nLevels2):
            # Corners
            j = j0 + i + 1
            j2 = i + 1 + (nSide2-1) - j0
            gmsh_to_ijk[nPts+0] = i + nSide * (j  + nSide * j)
            gmsh_to_ijk[nPts+1] = i + nSide * (j  + nSide * j2)
            gmsh_to_ijk[nPts+2] = i + nSide * (j2 + nSide * j2)
            gmsh_to_ijk[nPts+3] = i + nSide * (j2 + nSide * j)
            nPts += 4

            # Edges: Bottom, right, top, left
            nSide3 = nSide2 - 2 * (j0+1)
            for k in range(nSide3):
                gmsh_to_ijk[nPts+0*nSide3+k] = i + nSide * (j      + nSide * (j+1+k))
                gmsh_to_ijk[nPts+1*nSide3+k] = i + nSide * (j+1+k  + nSide * j2)
                gmsh_to_ijk[nPts+2*nSide3+k] = i + nSide * (j2     + nSide * (j2-1-k))
                gmsh_to_ijk[nPts+3*nSide3+k] = i + nSide * (j2-1-k + nSide * j)
            nPts += 4*nSide3

        # Center node for even-ordered Lagrange quads (odd value of nSide)
        if isOdd2:
            gmsh_to_ijk[nPts] = i + nSide * (nSide // 2 + nSide * (nSide // 2))
            nPts += 1

        # --- Right face ---
        for j0 in range(nLevels2):
            # Corners
            j = j0 + i + 1;
            j2 = i + 1 + (nSide2-1) - j0
            gmsh_to_ijk[nPts+0] = i2 + nSide * (j  + nSide * j)
            gmsh_to_ijk[nPts+1] = i2 + nSide * (j2 + nSide * j)
            gmsh_to_ijk[nPts+2] = i2 + nSide * (j2 + nSide * j2)
            gmsh_to_ijk[nPts+3] = i2 + nSide * (j  + nSide * j2)
            nPts += 4

            # Edges: Bottom, right, top, left
            nSide3 = nSide2 - 2 * (j0+1)
            for k in range(nSide3):
                gmsh_to_ijk[nPts+0*nSide3+k] = i2 + nSide * (j+1+k  + nSide * j)
                gmsh_to_ijk[nPts+1*nSide3+k] = i2 + nSide * (j2     + nSide * (j+1+k))
                gmsh_to_ijk[nPts+2*nSide3+k] = i2 + nSide * (j2-1-k + nSide * j2)
                gmsh_to_ijk[nPts+3*nSide3+k] = i2 + nSide * (j      + nSide * (j2-1-k))
            nPts += 4*nSide3

        # Center node for even-ordered Lagrange quads (odd value of nSide)
        if isOdd2:
            gmsh_to_ijk[nPts] = i2 + nSide * (nSide // 2 + nSide * (nSide // 2))
            nPts += 1

        # --- Back face ---
        for j0 in range(nLevels2):
            # Corners
            j = j0 + i + 1;
            j2 = i + 1 + (nSide2-1) - j0;
            gmsh_to_ijk[nPts+0] = j2 + nSide * (i2 + nSide * j)
            gmsh_to_ijk[nPts+1] = j  + nSide * (i2 + nSide * j)
            gmsh_to_ijk[nPts+2] = j  + nSide * (i2 + nSide * j2)
            gmsh_to_ijk[nPts+3] = j2 + nSide * (i2 + nSide * j2)
            nPts += 4

            # Edges: Bottom, right, top, left
            nSide3 = nSide2 - 2 * (j0+1)
            for k in range(nSide3):
                gmsh_to_ijk[nPts+0*nSide3+k] = j2-1-k + nSide * (i2 + nSide*j)
                gmsh_to_ijk[nPts+1*nSide3+k] = j      + nSide * (i2 + nSide*(j+1+k))
                gmsh_to_ijk[nPts+2*nSide3+k] = j+1+k  + nSide * (i2 + nSide*j2)
                gmsh_to_ijk[nPts+3*nSide3+k] = j2     + nSide * (i2 + nSide*(j2-1-k))
            nPts += 4*nSide3

        # Center node for even-ordered Lagrange quads (odd value of nSide)
        if isOdd2:
            gmsh_to_ijk[nPts] = nSide//2 + nSide * (i2 + nSide * (nSide//2))
            nPts += 1

        # --- Top face ---
        for j0 in range(nLevels2):
            # Corners
            j = j0 + i + 1
            j2 = i + 1 + (nSide2-1) - j0
            gmsh_to_ijk[nPts+0] = j  + nSide * (j  + nSide * i2)
            gmsh_to_ijk[nPts+1] = j2 + nSide * (j  + nSide * i2)
            gmsh_to_ijk[nPts+2] = j2 + nSide * (j2 + nSide * i2)
            gmsh_to_ijk[nPts+3] = j  + nSide * (j2 + nSide * i2)
            nPts += 4

            # Edges: Bottom, right, top, left
            nSide3 = nSide2 - 2 * (j0+1)
            for k in range(nSide3):
                gmsh_to_ijk[nPts+0*nSide3+k] = j+1+k  + nSide * (j      + nSide * i2)
                gmsh_to_ijk[nPts+1*nSide3+k] = j2     + nSide * (j+1+k  + nSide * i2)
                gmsh_to_ijk[nPts+2*nSide3+k] = j2-1-k + nSide * (j2     + nSide * i2)
                gmsh_to_ijk[nPts+3*nSide3+k] = j      + nSide * (j2-1-k + nSide * i2)
            nPts += 4*nSide3

        # Center node for even-ordered Lagrange quads (odd value of nSide)
        if isOdd2:
            gmsh_to_ijk[nPts] = nSide//2 + nSide * (nSide//2 +  nSide * i2)
            nPts += 1

    # Center node for even-ordered Lagrange quads (odd value of nSide)
    if isOdd:
        gmsh_to_ijk[nNodes-1] = nSide // 2 + nSide * (nSide // 2 + nSide * (nSide//2));

    return gmsh_to_ijk


class Py_callbacks(tg.callbacks):
    def __init__(self, system, griddata):
        tg.callbacks.__init__(self)

        self.system = system
        self.gid = system.gid
        self.griddata = griddata
        self.backend = system.backend
        tg.tioga_set_soasz(self.system.backend.soasz)
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

    # int* cellid int* nnodes # of solution points basis.upts
    def get_nodes_per_cell(self, cellid, nnodes):
        cidx = ptrAt(cellid,0)
        etype = self.griddata['celltypes'][cidx]
        n = self.system.ele_map[etype].basis.nupts
        writeAt(nnodes,0,int(n))

    # int* faceid, int *nnodes # fpt
    def get_nodes_per_face(self, faceid, nnodes):
        fidx = ptrAt(faceid,0)
        cidx1, cidx2  = self.griddata['f2corg'][fidx]
        fpos1, fpos2  = self.griddata['faceposition'][fidx] 
        etyp1  = self.griddata['celltypes'][cidx1]
        etyp2  = self.griddata['celltypes'][cidx2] if cidx2 >= 0 else etyp1
        n1 = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
        n2 = self.system.ele_map[etyp2].basis.nfacefpts[fpos2]
        if n1 != n2: raise RuntimeError('callback: nodes_per_face inconsistency')
        # write data to corresponding address
        writeAt(nnodes, 0, int(n1))

    # int* cellid, int* nnodes, double* xyz
    def get_receptor_nodes(self, cellid, nnodes, xyz):
        cidx = ptrAt(cellid,0)
        etyp = self.griddata['celltypes'][cidx]
        eles = self.system.ele_map[etyp]
        n = eles.basis.nupts
        writeAt(nnodes,0,n)
        if eles.mvgrid is True:
            ploc_upts = eles.ploc_at_ncon('upts')
        else:
            ploc_upts = eles.ploc_at('upts')
        # need local cell idx for this ele_map
        offset = self.griddata['celloffset'][cidx]
        typecid = cidx - offset
        coords = ploc_upts[:, :, typecid].reshape(-1)
        # write data to corresponding address
        for idx, coo in enumerate(coords):
            writeAt(xyz, idx, coo)

    # get the face points in standard element
    def get_face_nodes(self, faceid, nnodes, xyz):
        # fpts are calculate in elements
        fidx, nn = ptrAt(faceid,0), ptrAt(nnodes,0)
        cidx1, cidx2 = self.griddata['f2corg'][fidx]
        fpos1, fpos2 = self.griddata['faceposition'][fidx]
        etyp1  = self.griddata['celltypes'][cidx1]
        etyp2  = self.griddata['celltypes'][cidx2] if cidx2 >= 0 else etyp1
        if etyp1 != etyp2: raise RuntimeError('get_face_nodes: face type inconsistency')
        eles = self.system.ele_map[etyp1]
        fptsid = eles.basis.facefpts[fpos1]
        offset = self.griddata['celloffset'][cidx1]
        typecidx = cidx1 - offset
        plocfpts = eles.plocfpts[:,typecidx,:]
        fpts = plocfpts[fptsid].reshape(-1)
        for idx, coo in enumerate(fpts):
            writeAt(xyz, idx, coo)

    def donor_inclusion_test(self, cellid, xyz, passflag, rst):
        cidx = ptrAt(cellid,0)
        ndims = self.system.ndims
        # copy xyz into numpy array
        coords = np.array([ptrAt(xyz, i) for i in range(ndims)])[None,...]
        # using newton's method to get rst
        etype = self.griddata['celltypes'][cidx]
        eles = self.system.ele_map[etype]
        # shape coordinates 
        offset = self.griddata['celloffset'][cidx]
        typecid = cidx - offset
        eshape = eles.eles[:,cidx,:]
        # spts uniformly divided standard elements 
        stdshape = np.array(eles.basis.spts)
        stdlower = np.array([np.min(stdshape[:,i]) for i in range(ndims)])[None,...]
        stdupper = np.array([np.max(stdshape[:,i]) for i in range(ndims)])[None,...]

        rst_p = np.array([0.0,0.0,0.0])[None,...]
        rst_n = rst_p
        # using rst to get the initial xyz
        eop    = eles.basis.sbasis.nodal_basis_at(rst_p)

        dxyz = eop @ eshape - coords
        # apply newton's method
        iters = 0
        while np.linalg.norm(dxyz)>1e-10:
            # get the jac_nodal_basis
            eop    = eles.basis.sbasis.nodal_basis_at(rst_n)
            eop_df = eles.basis.sbasis.jac_nodal_basis_at(rst_n).swapaxes(1,2)

            rst_n = rst_n -  np.dot((eop @ eshape - coords),
                             np.linalg.inv((eop_df @ eshape).reshape(ndims,ndims)))

            dxyz = eop @ eshape - coords
            iters = iters + 1
            if iters > 10:
                break

        if np.linalg.norm(dxyz) <= 1e-6: # converged
            dd = (rst_n > stdlower - 1e-5)
            ee = (rst_n < stdupper + 1e-5)
            ff = np.logical_and(dd,ee)
            if np.all(ff):
                writeAt(passflag,0,1) # pass
            else:
                writeAt(passflag,0,0) # no pass
        else:
            writeAt(passflag,0,0) # no pass
        for i in range(ndims):
            writeAt(rst,i,rst_n[0][i])

    def convert_to_modal(self, cellid, nspts, q_in, npts, index_out, q_out):
        cid, nnspts = ptrAt(cellid,0), ptrAt(nspts,0)
        idout = cid*nnspts
        writeAt(index_out,0,idout)
        for i in range(nnspts):
            writeAt(q_out,i,ptrAt(q_in[i]))

    def donor_frac(self, cellid, xyz, nweights, inode, weights, rst, buffsize):
        # inode is not used xyz is not used
        cidx = ptrAt(cellid,0)
        ndims = self.system.ndims
        etype = self.griddata['celltypes'][cidx]
        eles  = self.system.ele_map[etype]
        nspts = eles.basis.nupts
        writeAt(nweights,0,nspts)
        assert nspts <= ptrAt(buffsize,0), 'weights buffer not enough space'
        offset = self.griddata['celloffset'][cidx]
        typecidx = cidx - offset
        rst_loc = np.array([ptrAt(rst, i) for i in range(ndims)])[None,...]
        eop = eles.basis.ubasis.nodal_basis_at(rst_loc)
        for i in range(nspts):
            writeAt(weights,i,eop[i])
    
    def donor_frac_gpu(self, cellids, nfringe, rst, weights):
        # tioga does not support multiple type of elements
        etype = 'hex-g{}'.format(self.gid)
        eles = self.system.ele_map[etype]
        order = eles.basis.order
        nSpts1D = order+1
        nspts = eles.basis.nupts
        spts1d = np.atleast_2d(
            np.array(eles.basis.upts[:order+1,0])
        ).astype(self.backend.fpdtype)
        # copy data to device
        xiGrid = self.griddata['xi1d']
        tg.tg_copy_to_device(
          xiGrid, addrToFloatPtr(spts1d.__array_interface__['data'][0]), spts1d.nbytes
        )
        tg.get_nodal_basis_wrapper(
          cellids,rst,weights,addrToFloatPtr(xiGrid),nfringe,int(nspts),int(nSpts1D),3
        )

    # using golfbal cellidx to get the ele_map as well as local cidx per type
    def find_eles_instance(self, cellid):
        etype = self.griddata['celltypes'][cellid]
        eles = self.system.ele_map[etype]
        offset = self.griddata['celloffset'][cellid]
        cidx = cellid - offset
        return eles, cidx
    
    # return a float 
    def get_q_spt(self, cidx, spt, var):
        eles, typecidx = self.find_eles_instance(cidx)
        elesdata = eles.scal_upts_inb._curr_mat
        # check if it is an numpy array
        if hasattr(elesdata,'__array_interface__'):
            # using etype and eid to calculate the offset
            datashape = elesdata.datashape # [nspt, neled1, nvar, neled2]
            neled1, neled2 = datashape[1], datashape[-1]
            neles, nvars = eles.neles, eles.nvars
            offset = ((typecidx//neled2)*nvars + var)*nelesd2 + typecidx % nelesd2
            # calculate the datashape
            a = elesdata.data[spt,offset]
        else:
            # using etype and eid to calculate the offset
            raise RuntimeError('CUDA get_q_spt not implemented')
        return a

    # return an address
    def get_q_fpt(self, fidx, fpt, var):
        # note data are mapped 
        cidx1, cidx2 = self.griddata['f2corg'][fidx]
        # use left or right for interior artificial bnds? that's a question
        # using blanking info to choose whether left or right
        side = 0 if self.griddata['iblank_cell'][cidx1] == 1 else 1
        cidx_f = cidx1 if side == 0 else cidx2
        if cidx_f < 0: raise RuntimeError("get_q_fpt, cidx_f <0")
        eles, typecidx = self.find_eles_instance(cidx_f)
        elesdata = eles._scal_fpts

        if hasattr(elesdata.data, '__array_interface__'):
            datashape = elesdata.datashape
            nfpt, neled1, nvars, neled2 = datashape
            offset = ((typecidx//neled2)*nvars + var)*neled2 + typecidx % neled2
            a = elesdata.data[fpt:, offset:]
        else:
            raise RuntimeError('CUDA get_q_fpt not implemented')

        # here return the address as int
        return a.__array_interface__['data'][0]

    # return a float
    def get_dq_spt(self, cidx, spt, dim, var):
        eles, typecidx = self.find_eles_instance(cidx)
        elesdata = eles._vect_upts
        if hasattr(elesdata.data,'__array_interface__'):
            # using etype and eid to calculate the offset
            datashape = elesdata.datashape # [dim, nspt, neled1, nvar, neled2]
            dim, nspt = datashape[0], datashape[1]
            neled1, neled2 = datashape[2], datashape[-1]
            neles, nvars = eles.neles, eles.nvars
            offset = ((typecidx//neled2)*nvars + var)*neled2 + typecidx % neled2
            # calculate the datashape
            a = elesdata.data[dim*nspt+spt,offset]
        else:
            # using etype and eid to calculate the offset
            raise RuntimeError('CUDA get_q_spt not implemented')
        return a

    # return an address
    def get_dq_fpt(self, fidx, fpt, dim,var):
        cidx1, cidx2 = self.griddata['f2corg'][fidx]
        side = 0 if self.griddata['iblank_cell'][cidx1] == 1 else 1
        cidx_f = cidx1 if side == 0 else cidx2
        if cidx_f < 0: raise RuntimeError("get_dq_fpt, cidx_f <0")
        eles, typecidx = self.find_eles_instance(cidx_f)

        elesdata = eles._vect_fpts
        # here return the address as int
        if hasattr(elesdata, '__array_interface__'):
            datashape = elesdata.datashape
            ndims, nfpt, neled1, nvars, neled2 = datashape
            offset = ((typecidx//neled2)*nvars + var)*neled2 + typecidx % neled2
            a = elesdata.data[ (nfpt*dim + fpt):, offset:]
        else:
            raise RuntimeError('CUDA get_q_fpt not implemented')
        return a.__array_interface__['data'][0]

    # return an address
    def get_q_spts(self, ele_stride, spt_stride, var_stride, ele_type):
        etype = 'hex-g{}'.format(self.gid)
        eles = self.system.ele_map[etype]
        elesdata = eles.scal_upts_inb._curr_mat
        nspt, neled1, nvars, neled2 = elesdata.datashape
        es = nvars*neled2
        ss = neled1*nvars*neled2
        vs = neled2
        writeAt(ele_stride, 0, es)
        writeAt(spt_stride, 0, ss)
        writeAt(var_stride, 0, vs)
        return elesdata.data.__array_interface__['data'][0]

    # return an address
    def get_dq_spts(self, ele_stride, spt_stride, var_stride, dim_stride, ele_type):
        etype = 'hex-g{}'.format(self.gid)
        eles = self.system.ele_map[etype]
        elesdata = eles._vect_upts
        ndim, nspt, neled1, nvars, neled2 = elesdata.datashape
        es =  nvars*neled2
        ss =  neled1*nvars*neled2 
        vs =  neled2
        ds =  neled1*nvars*neled2*nspt
        writeAt(ele_stride, 0, es)
        writeAt(spt_stride, 0, ss)
        writeAt(var_stride, 0, vs)
        writeAt(dim_stride, 0, ds)
        return elesdata.data.__array_interface__['data'][0]
    
    # fringe_data_to_device
    # see faces.cpp

    def ff(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        
        if self.rank==1:
            lhs=self.system._mpi_inters[0].lhs
            fcc=[]
            for i in lhs:
                cc=(i[1],i[2])
                fcc.append(cc)
            #print(f'{self.rank=}',f'{self.nfringe=}')
            #print(self.fringe_faceinfo)
            #matrix_entry = self._scal_view_fpts_ploc(
                #self.fringe_faceinfo,'get_scal_unsrted_fpts_ploc_for_inter')
            #print(matrix_entry._mats[0].get().shape)
            #exit()
            #print(f'{dir(self.system)=}')      
            ss='hex-g0'
            #print( f'{(self.system.ele_map[ss].plocfpts.shape)=}','\n')
            #print( f'{len(self.system.ele_map[ss]._srtd_face_fpts[0])=}','\n')
            #print( f'{self.system._mpi_inters[0].lhs}','\n')
            Ploc=self.system.ele_map[ss].plocfpts
            nnl=[]
            for idx,i in enumerate(fcc):
                a=i[0]
                b=i[1]
                print(a)
                nn=Ploc[4*b:4*b+4,a,:]
                nnl.append(nn)
            #print(nnl)                

            #exit()
            #print(plocfpts)
            for idx,m in enumerate(self.system._mpi_inters):

                #print(dir(self.system))
                #scalrhs = self._scal_view(m.rhs, 'get_scal_fpts_for_inter')
                AAA=m._scal_rhs.get()
                BBB=m._vect_rhs.get()
                print(idx,BBB)
                name=f'foo-{idx}.dat'
                np.savetxt(name, BBB[1], delimiter=",")
                
                #print(idx,dir(m))
                #sys.exit()
                #c=addrToFloatPtr(int(m._scal_rhs.data))
                #tg.tg_print_data(m._scal_rhs.data)
                #tg.tg_print_data(int(m.),0,1200,2)
            
            
                #datatest=ptrToArray(m._scal_rhs,12*5)
            #for idx, (etype, eles) in enumerate(self.system.ele_map.items()):
            #    print(eles.eles.reshape(-1,self.system.ndims))
            sys.exit()
    def fringe_data_to_device(self, fringeids, nfringe, gradflag, data):
        if nfringe == 0: return
        if gradflag == 0: # passing u
            self.fringe_u_device(fringeids, nfringe, data)
        else: # passing du
            self.fringe_du_device(fringeids, nfringe, data)

    def fringe_u_device(self, fringeids, nfringe, data):
        if nfringe == 0: return 0

        self.tot_nfpts_mpi=0
        self.tot_nfpts_fringe=0

        # First deal with  inner artbnd
        tot_nfpts = 0
        faceinfo = []
        side_int = []
        side_idx = [0]
        faceinfo_test = []
        
        facedata = []
        facefringeid = []
        facenfpts = []

        kk=0

        for i in range(nfringe):
            fid = ptrAt(fringeids,i)
            cidx1, cidx2 = self.griddata['f2corg'][fid]
            fpos1, fpos2 = self.griddata['faceposition'][fid]
            etyp1 = self.griddata['celltypes'][cidx1]
            etyp2 = self.griddata['celltypes'][cidx2] 

            
            if cidx2 > 0:
                

                side = 1 if self.griddata['iblank_cell'][cidx1] == 1 else 0
                # for multiple element types in one partition
                if self.griddata['iblank_cell'][cidx1] == self.griddata['iblank_cell'][cidx2] :
                    raise RuntimeError("this should not happen")
                perface = (
                    (etyp1, cidx1 - self.griddata['celloffset'][cidx1], fpos1, 0) 
                    if side == 0 else 
                    (etyp2, cidx2 - self.griddata['celloffset'][cidx2], fpos2, 0)
                )

                facefringeid.append(i)

                faceinfo.append(perface)
                side_int.append(side)
                
                # always use left face
                nfpts = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
                tot_nfpts = tot_nfpts + nfpts
                side_idx.append(tot_nfpts)
                
                # Grab the data
                facedata = facedata + [ptrAt(data,i*self.system.nvars*nfpts+j) 
                                       for j in range(self.system.nvars*nfpts)]
                facenfpts.append(nfpts)

        # save for later use
        self.fringe_u_fpts_d = self.griddata['fringe_u_fpts_d']
        self.fringe_faceinfo = faceinfo
        self.tot_nfpts_fringe = tot_nfpts
        self.facefringeid = facefringeid
        self.facenfpts = facenfpts
        self.faceinfo=faceinfo
        
        #print(scal_fpts_ploc)


        if faceinfo != []:
            self._scal_fpts_u = self._scal_view_fpts_u(
                faceinfo, 'get_scal_unsrted_fpts_for_inter')
            
            nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts
            nbytes = nbytes*self.system.nvars

            facedata = np.array(facedata).astype(self.backend.fpdtype)

            tg.tg_copy_to_device( 
                self.fringe_u_fpts_d, 
                addrToFloatPtr(facedata.__array_interface__['data'][0]), 
                int(nbytes)
            )
            
            # datashape of eles._scal_fpts is [nfpts, neled2, nvars, soasz]
            tg.unpack_fringe_u_wrapper (
                addrToFloatPtr(self.fringe_u_fpts_d),
                addrToFloatPtr(int(self._scal_fpts_u._mats[0].basedata)),
                addrToUintPtr(self._scal_fpts_u.mapping.data),
                nfringe, tot_nfpts, self.system.nvars, self.backend.soasz, 3
            )
        
        # Then deal with overset artbnd
        tot_nfpts_ov = 0
        faceinfo_ov = []
        facefpts_ov_range = [0]

        facedata_ov = []
        facefringeid_ov = []
        facenfpts_ov = []
        for i in range(nfringe):
            fid = ptrAt(fringeids,i)
            cidx1, cidx2 = self.griddata['f2corg'][fid]
            fpos1, fpos2 = self.griddata['faceposition'][fid]
            etyp1 = self.griddata['celltypes'][cidx1]
            etyp2 = self.griddata['celltypes'][cidx2] 
            
            if cidx2 < 0 and cidx2 != -2:
                #if cidx2 == -2: print(f'{cidx2} for {fid}')
                # always use left info here
                #print(f'{self.rank=}',cidx2)
                perface = (etyp1, cidx1 - self.griddata['celloffset'][cidx1], fpos1, 0) 
                faceinfo_ov.append(perface)

                facefringeid_ov.append(i)
                # always use left face
                nfpts = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
                tot_nfpts_ov = tot_nfpts_ov + nfpts
                facefpts_ov_range.append(tot_nfpts_ov)
                # Grab the data
                facedata_ov = facedata_ov + [ptrAt(data,i*self.system.nvars*nfpts+j) 
                                             for j in range(self.system.nvars*nfpts)]
                facenfpts_ov.append(nfpts)
        # save for later use
        self.fringe_faceinfo_ov = faceinfo_ov
        # save for later use
        self.tot_nfpts_ov = tot_nfpts_ov

        self.facefringeid_ov = facefringeid_ov
        self.facenfpts_ov = facenfpts_ov

        if faceinfo_ov != []:

            nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts_ov
            nbytes = nbytes*self.system.nvars
            
            facedata_ov = np.array(facedata_ov).astype(self.backend.fpdtype)
            facedata_ov = facedata_ov.reshape(-1,self.system.nvars)

            for idx, face in enumerate(faceinfo_ov):
                etype, typecidx, fpos, _ = face
                srted_ord = self.system.ele_map[etype]._srtd_face_fpts[fpos][typecidx]
                unsrted_ord = self.system.ele_map[etype].basis.facefpts[fpos]
                # swap data according to the node ordering
                nodesmap = []
                for n in srted_ord:
                    nodesmap.append(unsrted_ord.index(n))

                # swap data 
                a = facedata_ov[facefpts_ov_range[idx]:facefpts_ov_range[idx+1]]
                facedata_ov[facefpts_ov_range[idx]:facefpts_ov_range[idx+1]] = a[nodesmap]

            cc = facedata_ov.swapaxes(0,1).reshape(-1)

            matrix_entry = self.system._mpi_inters[-1]._scal_rhs

            tg.tg_copy_to_device(
                matrix_entry.data,
                addrToFloatPtr(cc.ctypes.data),
                int(nbytes)
            )

        
        for midx, mpiinters in enumerate(self.system._mpi_inters):
            # then deal with overset artbnd
            scal_size= len(mpiinters.lhs)

            tot_nfpts_mpi = 0
            faceinfo_mpi = []
            facefpts_mpi_range = [0]
            facefringeid_mpi = []
            facedata_mpi = []
            facenfpts_mpi = []
            for i in range(nfringe):
                fid = ptrAt(fringeids,i)
                cidx1, cidx2 = self.griddata['f2corg'][fid]
                fpos1, fpos2 = self.griddata['faceposition'][fid]
                etyp1 = self.griddata['celltypes'][cidx1]
                etyp2 = self.griddata['celltypes'][cidx2] 
            
                if cidx2 == -2 and self.griddata['face_inters_idx'][fid] == midx:
                    # always use left info here
                    perface = (etyp1, cidx1 - self.griddata['celloffset'][cidx1], fpos1, 0) 
                    faceinfo_mpi.append(perface)
                    facefringeid_mpi.append(i)

                    # always use left face
                    nfpts = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
                    tot_nfpts_mpi = tot_nfpts_mpi + nfpts
                    facefpts_mpi_range.append(tot_nfpts_mpi)
                    # grab data
                    facedata_mpi = facedata_mpi + [ptrAt(data,i*self.system.nvars*nfpts+j) 
                                             for j in range(self.system.nvars*nfpts)]

                    facenfpts_mpi.append(nfpts)

            # save for later use
            self.fringe_faceinfo_mpi = faceinfo_mpi
            # save for later use

            self.tot_nfpts_mpi += tot_nfpts_mpi

            self.facenfpts_mpi = facenfpts_mpi
               
            if faceinfo_mpi != []:
                
                facedata_mpi = np.array(facedata_mpi).astype(self.backend.fpdtype)
                facedata_mpi = facedata_mpi.reshape(-1,self.system.nvars)

                facedata_mpi_size=facedata_mpi.shape
                
                

                for idx, face in enumerate(faceinfo_mpi):
                    etype, typecidx, fpos, _ = face
                    srted_ord = self.system.ele_map[etype]._srtd_face_fpts[fpos][typecidx]
                    unsrted_ord = self.system.ele_map[etype].basis.facefpts[fpos]
                    # swap data according to the node ordering
                    nodesmap = []
                    for n in srted_ord:
                        nodesmap.append(unsrted_ord.index(n))

                    #if self.rank==1:
                        ##print(self.system.ele_map[etype])
                    #nodesmap=[1,3,0,2]
                    # swap data 
                    a = facedata_mpi[facefpts_mpi_range[idx]:facefpts_mpi_range[idx+1]]
                    #facedata_mpi[facefpts_mpi_range[idx]:facefpts_mpi_range[idx+1]] = a[nodesmap]

                    # Find the index of the face in all mpi faces
                    startIdx = mpiinters.lhs.index(face)

                    dataperface = a[nodesmap]
                    # Do the copy face by face instead of together


                    
                    cc = dataperface.swapaxes(0,1).reshape(-1)
                    
                    

                    #exit()
                    # overset-mpi is the last one
                    matrix_entry = mpiinters._scal_rhs
                    
                    # Need to figure out the locations of these faces in the matrix entry
                    nfpts = facenfpts_mpi[idx]
                    nvars=self.system.nvars
                    for i in range(5):
                        cc1=cc[i*nfpts:(i+1)*nfpts]
                        nbytes = np.dtype(self.backend.fpdtype).itemsize*nfpts
                        #nbytes = nbytes*self.system.nvars
                        # calculate the offset of this face in the matrix entry
                        offset = startIdx*nbytes+i*scal_size*nfpts*np.dtype(self.backend.fpdtype).itemsize
                        #if self.rank==0:
                        #    print(midx,f'{mpiinters._rhsrank=}',f'{idx=}',f'{self.rank=}',f'{cc1=}',f'{startIdx=}',f'{offset=}')

                        #tg.tg_copy_to_device(matrix_entry.data,data,int(nbytes))
                        
                        tg.tg_copy_to_device(
                            matrix_entry.data+offset,
                            addrToFloatPtr(cc1.ctypes.data),
                            int(nbytes)
                        )                        
                        
                    #exit()



















    def fringe_du_device(self, fringeids, nfringe, data):
        
        if self.tot_nfpts_fringe+self.tot_nfpts_mpi==0:
            return 0
        if self.tot_nfpts_fringe>0:
            tot_nfpts=self.tot_nfpts_fringe
            # Deal with interior artificial boundary conditions
            faceinfo = self.fringe_faceinfo
            # copy data to device
            self.fringe_du_fpts_d = self.griddata['fringe_du_fpts_d']
            nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts
            nbytes = nbytes*self.system.nvars*self.system.ndims

            


            facedata = []

            for i, nfpts in zip(self.facefringeid, self.facenfpts):
                facedata = facedata + [ptrAt(data,i*self.system.nvars*self.system.ndims*nfpts+j) 
                                for j in range(self.system.nvars*self.system.ndims*nfpts)]
                
            facedata = np.array(facedata).astype(self.backend.fpdtype)
            ##############

            tg.tg_copy_to_device(
                self.fringe_du_fpts_d, 
                addrToFloatPtr(facedata.__array_interface__['data'][0]),
                int(nbytes)
            )

            self._vect_fpts_du = self._vect_view_fpts_du(
                    faceinfo, 'get_vect_unsrted_fpts_for_inter')
            # copy to the place
            matrix_entry = self.system.ele_map['hex-g{}'.format(self.gid)]._vect_fpts
            datashape = matrix_entry.datashape
            dim_stride = datashape[1]*datashape[2]*datashape[3]*datashape[4]
            # datashape of eles._scal_fpts is [nfpts, neled2, nvars, soasz]
            tg.unpack_fringe_grad_wrapper (
                addrToFloatPtr(self.fringe_du_fpts_d),
                addrToFloatPtr(int(matrix_entry.basedata)),
                addrToUintPtr(self._vect_fpts_du.mapping.data),
                addrToUintPtr(self._vect_fpts_du.rstrides.data),
                nfringe, tot_nfpts, self.system.nvars, self.system.ndims,
                self.backend.soasz, 3
            )
        if self.tot_nfpts_mpi>0:
                
            for midx, mpiinters in enumerate(self.system._mpi_inters):
                scal_size= len(mpiinters.lhs)
                # then deal with overset artbnd
                tot_nfpts_mpi = 0
                faceinfo_mpi = []
                facefpts_mpi_range = [0]
                facefringeid_mpi = []
                facedata_mpi = []
                facenfpts_mpi = []
                for i in range(nfringe):
                    fid = ptrAt(fringeids,i)
                    cidx1, cidx2 = self.griddata['f2corg'][fid]
                    fpos1, fpos2 = self.griddata['faceposition'][fid]
                    etyp1 = self.griddata['celltypes'][cidx1]
                    etyp2 = self.griddata['celltypes'][cidx2] 
                
                    if cidx2 == -2 and self.griddata['face_inters_idx'][fid] == midx:
                        # always use left info here
                        perface = (etyp1, cidx1 - self.griddata['celloffset'][cidx1], fpos1, 0) 
                        faceinfo_mpi.append(perface)
                        facefringeid_mpi.append(i)

                        # always use left face
                        nfpts = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
                        tot_nfpts_mpi = tot_nfpts_mpi + nfpts
                        facefpts_mpi_range.append(tot_nfpts_mpi)

                        # Grab data for gradient
                        facedata_mpi = facedata_mpi + [ptrAt(data,i*self.system.nvars*self.system.ndims*nfpts+j) 
                            for j in range(self.system.nvars*self.system.ndims*nfpts)]

                        facenfpts_mpi.append(nfpts)
                if faceinfo_mpi != []:
                    
                    facedata_mpi = np.array(facedata_mpi).astype(self.backend.fpdtype)
                    facedata_mpi = facedata_mpi.reshape(-1, self.system.ndims, self.system.nvars)
                    ## for test
                    facedata_mpi_size=facedata_mpi.shape
                    #################
                    #facedata_mpi=np.zeros(facedata_mpi_size)
                    #facedata_mpi=np.ones(facedata_mpi_size)
                    #print(f'{facedata_mpi=}')

                #################
                    for idx, face in enumerate(faceinfo_mpi):
                        etype, typecidx, fpos, _ = face
                        srted_ord = self.system.ele_map[etype]._srtd_face_fpts[fpos][typecidx]
                        unsrted_ord = self.system.ele_map[etype].basis.facefpts[fpos]
                        # swap data according to the node ordering
                        nodesmap = []
                        for n in srted_ord:
                            nodesmap.append(unsrted_ord.index(n))
                        # swap data 
                        a = facedata_mpi[facefpts_mpi_range[idx]:facefpts_mpi_range[idx+1]]
                        #facedata_mpi[facefpts_mpi_range[idx]:facefpts_mpi_range[idx+1]] = a[nodesmap]
                        # Find the index of the face in all mpi faces
                        startIdx = mpiinters.lhs.index(face)

                        dataperface = a[nodesmap]
                        # Do the copy face by face instead of together
                        cc = dataperface.swapaxes(0,1)
                        # do the copy dimension by dimension
                        matrix_entry = mpiinters._vect_rhs
                        for idim in range(self.system.ndims):
                            # overset-mpi is the last one
                            
                            datashape = matrix_entry.datashape
                            
                            # Need to figure out the locations of these faces in the matrix entry
                            nfpts = facenfpts_mpi[idx]
                            nbytes = np.dtype(self.backend.fpdtype).itemsize*nfpts
                            nbytes = nbytes*self.system.nvars
                            # calculate the offset of this face in the matrix entry
                            
                             

                            ccf = cc[idim].swapaxes(0,1).reshape(-1)
                            
                            nvars=self.system.nvars
                            for i in range(nvars):
                                cc1=ccf[i*nfpts:(i+1)*nfpts]
                                nbytes = np.dtype(self.backend.fpdtype).itemsize*nfpts
                                #nbytes = nbytes*self.system.nvars
                                # calculate the offset of this face in the matrix entry
                                offset = startIdx*nbytes+(i*scal_size*nfpts+idim*datashape[1])*np.dtype(self.backend.fpdtype).itemsize
                                tg.tg_copy_to_device(
                                    matrix_entry.data+offset,
                                    addrToFloatPtr(cc1.ctypes.data),
                                    int(nbytes)
                                )
                    








        for midx, mpiinters in enumerate(self.system._mpi_inters):
            # then deal with overset artbnd
            tot_nfpts_mpi = 0
            faceinfo_mpi = []
            facefpts_mpi_range = [0]
            facefringeid_mpi = []
            facedata_mpi = []
            facenfpts_mpi = []
            for i in range(nfringe):
                fid = ptrAt(fringeids,i)
                cidx1, cidx2 = self.griddata['f2corg'][fid]
                fpos1, fpos2 = self.griddata['faceposition'][fid]
                etyp1 = self.griddata['celltypes'][cidx1]
                etyp2 = self.griddata['celltypes'][cidx2] 
            
                if cidx2 == -2 and self.griddata['face_inters_idx'][fid] == midx:
                    # always use left info here
                    perface = (etyp1, cidx1 - self.griddata['celloffset'][cidx1], fpos1, 0) 
                    faceinfo_mpi.append(perface)
                    facefringeid_mpi.append(i)

                    # always use left face
                    nfpts = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
                    tot_nfpts_mpi = tot_nfpts_mpi + nfpts
                    facefpts_mpi_range.append(tot_nfpts_mpi)

                    # Grab data for gradient
                    facedata_mpi = facedata_mpi + [ptrAt(data,i*self.system.nvars*self.system.ndims*nfpts+j) 
                           for j in range(self.system.nvars*self.system.ndims*nfpts)]

                    facenfpts_mpi.append(nfpts)

            if faceinfo_mpi != []:
                
                facedata_mpi = np.array(facedata_mpi).astype(self.backend.fpdtype)
                facedata_mpi = facedata_mpi.reshape(-1, self.system.ndims, self.system.nvars)
                
                for idx, face in enumerate(faceinfo_mpi):
                    etype, typecidx, fpos, _ = face
                    srted_ord = self.system.ele_map[etype]._srtd_face_fpts[fpos][typecidx]
                    unsrted_ord = self.system.ele_map[etype].basis.facefpts[fpos]
                    # swap data according to the node ordering
                    nodesmap = []
                    for n in srted_ord:
                        nodesmap.append(unsrted_ord.index(n))
                    # swap data 
                    a = facedata_mpi[facefpts_mpi_range[idx]:facefpts_mpi_range[idx+1]]
                    facedata_mpi[facefpts_mpi_range[idx]:facefpts_mpi_range[idx+1]] = a[nodesmap]
                    # Find the index of the face in all mpi faces
                    startIdx = mpiinters.lhs.index(face)

                    dataperface = a[nodesmap]
                    # Do the copy face by face instead of together
                    cc = dataperface.swapaxes(0,1)
                    # do the copy dimension by dimension
                    
                    for idim in range(self.system.ndims):
                        # overset-mpi is the last one
                        matrix_entry = mpiinters._vect_rhs

                        datashape = matrix_entry.datashape
                        
                        # Need to figure out the locations of these faces in the matrix entry
                        nfpts = facenfpts_mpi[idx]
                        nbytes = np.dtype(self.backend.fpdtype).itemsize*nfpts
                        nbytes = nbytes*self.system.nvars
                        # calculate the offset of this face in the matrix entry
                        offset = (startIdx)*nbytes + datashape[1]*np.dtype(self.backend.fpdtype).itemsize

                        ccf = cc[idim].swapaxes(0,1).reshape(-1)
                        
                        tg.tg_copy_to_device(
                            matrix_entry.data+offset,
                            addrToFloatPtr(ccf.ctypes.data),
                            int(nbytes)
                        )
                
    def get_face_nodes_gpu(self, faceids, nfaces, nptsface, xyz):
        # these faces are 
        if nfaces == 0: return
        tot_nfpts = 0
        for i in range(nfaces):
            tot_nfpts = tot_nfpts + ptrAt(nptsface, i)
        # first build up the interface information
        faceinfo = []
        for i in range(nfaces):
            fid = ptrAt(faceids,i)
            # using faceid to get the nfpts
            cidx1, cidx2 = self.griddata['f2corg'][fid]
            if cidx2 <0: cidx2 = cidx1
            fpos1, fpos2 = self.griddata['faceposition'][fid]
            etyp1    = self.griddata['celltypes'][cidx1]
            etyp2    = self.griddata['celltypes'][cidx2] 
            
            perface = (etyp1, cidx1 - self.griddata['celloffset'][cidx1], fpos1, 0)
            faceinfo.append(perface)

        # note there need to use unsorted face idx
        # check base/element.py
        self._scal_fpts_ploc = self._scal_view_fpts_ploc(
                faceinfo, 'get_scal_unsrted_fpts_ploc_for_inter')
        
        self.fringe_coords_d = self.griddata['fringe_coords_d']
        # datashape of ploc_at_ncon('fpts') is [nfpts, neled2, dim, soasz]
        # pointwise operation
        tg.pack_fringe_coords_wrapper(
            addrToUintPtr(self._scal_fpts_ploc.mapping.data), # offset PyFR martix
            addrToFloatPtr(self.fringe_coords_d), # a flat matrx 
            addrToFloatPtr(int(self._scal_fpts_ploc._mats[0].basedata)),# starting addrs
            tot_nfpts, self.system.ndims, self.backend.soasz, 3
        )
        
        nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts
        nbytes = nbytes*self.system.ndims
        tg.tg_copy_to_host(self.fringe_coords_d, xyz, nbytes)

    # cell data used for unblanking, infuse data into unblanked cells
    # unblank_to_device
    def cell_data_to_device(self, cellids, ncells, gradflag, data):
        if ncells == 0: return 
        if gradflag == 0:
            self.cell_u_to_device(cellids, ncells, data)
        else:
            # this will never be called 
            self.cell_du_to_device(cellids, ncells, data)

    def cell_u_to_device(self, cellids, ncells, data):
        # there are from get_cell_coords
        unblank_ids_ele = self.unblank_ids_ele_host
        unblank_ids_loc = self.unblank_ids_loc_host
        self.unblank_u_d = self.griddata['unblank_u_d']

        # copy data from host to device
        tot_nspts = self.tot_nspts
        nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nspts*self.system.nvars
        tg.tg_copy_to_device(self.unblank_u_d, data, int(nbytes))

        # for each type do the copy inject data into unblanked cells
        for etype in unblank_ids_ele.keys():
            ele_soln = self.system.ele_map[etype].scal_upts_inb._curr_mat
            nspts = self.system.ele_map[etype].basis.nupts
            encells = unblank_ids_ele[etype].shape[1]
            
            neled2 = ele_soln.datashape[1]

            tg.unpack_unblank_u_wrapper(
                addrToIntPtr(self.unblank_ids_loc[etype]),
                addrToIntPtr(self.unblank_ids_ele[etype]),
                addrToFloatPtr(self.unblank_u_d),
                addrToFloatPtr(ele_soln.data),
                int(encells), int(nspts), int(self.system.nvars), 
                int(self.system.backend.soasz), int(neled2), 3
            )

    def cell_du_to_device(self, cellids, ncell, data):
        raise RuntimeError("copy du to device should not happen")
    
    def get_cell_nodes_gpu(self, cellids, ncells, nptscell, xyz):
        if ncells == 0 : return
        unblank_ids_ele = OrderedDict()
        unblank_ids_loc = OrderedDict() # stores the entrance of each cell spts
        for etype in self.system.ele_types:
            unblank_ids_ele[etype] = []
            unblank_ids_loc[etype] = []
        # do the copy element by element
        tot_nspts = 0
        for i in range(ncells):
            # cell id
            cid = ptrAt(cellids, i)
            etype = self.griddata['celltypes'][cid]
            typecidx = self.griddata['celloffset'][cid]
            unblank_ids_ele[etype].append(cid-typecidx) 
            npts = self.system.ele_map[etype].basis.nupts
            unblank_ids_loc[etype].append(tot_nspts)
            tot_nspts = tot_nspts + npts
            # write data to nptscell
            writeAt(nptscell, i, int(npts))
        for etype in self.system.ele_types:
            unblank_ids_ele[etype] = np.atleast_2d(
                np.array(unblank_ids_ele[etype])
            ).astype('int32')
            unblank_ids_loc[etype] = np.atleast_2d(
                np.array(unblank_ids_loc[etype])
            ).astype('int32')
        
        # save for later use
        self.unblank_ids_ele_host = unblank_ids_ele
        self.unblank_ids_loc_host = unblank_ids_loc
        self.tot_nspts = tot_nspts
        self.unblank_ids_ele = self.griddata['unblank_ids_ele']
        self.unblank_ids_loc = self.griddata['unblank_ids_loc']
        self.unblank_coords_d = self.griddata['unblank_coords_d']

        # copy unblank_ids_ele and unblank_ids_loc to backend
        for etype in unblank_ids_ele.keys():
            ptr_ele = unblank_ids_ele[etype].__array_interface__['data'][0]
            ptr_loc = unblank_ids_loc[etype].__array_interface__['data'][0]
            nbytes_e = unblank_ids_ele[etype].nbytes
            nbytes_l = unblank_ids_loc[etype].nbytes
            tg.tg_copy_to_device(
                self.unblank_ids_ele[etype],addrToIntPtr(ptr_ele), int(nbytes_e)
            )
            tg.tg_copy_to_device(
                self.unblank_ids_loc[etype],addrToIntPtr(ptr_loc), int(nbytes_l)
            )

        # for each type do the copy
        for etype in unblank_ids_ele.keys():
            ele_coords = self.system.ele_map[etype].ploc_at_ncon('upts')
            nspts = self.system.ele_map[etype].basis.nupts
            ncells = unblank_ids_ele[etype].shape[1]
            neled2 = ele_coords.datashape[1]

            tg.pack_cell_coords_wrapper(
                addrToIntPtr(self.unblank_ids_loc[etype]),
                addrToIntPtr(self.unblank_ids_ele[etype]),
                addrToFloatPtr(self.unblank_coords_d),
                addrToFloatPtr(ele_coords.data),
                int(ncells), int(nspts), int(self.system.ndims), 
                int(self.system.backend.soasz), int(neled2), 3
            )
        # copy data from device to xyz
        nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nspts*self.system.ndims
        tg.tg_copy_to_host(self.unblank_coords_d, xyz, int(nbytes))
        
    def get_q_spts_gpu(self, ele_stride, spt_stride, var_stride, ele_type):
        etype = 'hex-g{}'.format(self.gid)
        eles = self.system.ele_map[etype]
        elesdata = eles.scal_upts_inb._curr_mat
        nspt, neled1, nvars, neled2 = elesdata.datashape
        es = nvars*neled2
        ss = neled1*nvars*neled2
        vs = neled2
        writeAt(ele_stride, 0, es)
        writeAt(spt_stride, 0, ss)
        writeAt(var_stride, 0, vs)
        return int(elesdata.data)
        
    def get_dq_spts_gpu(self, ele_stride, spt_stride, var_stride, dim_stride, ele_type):
        etype = 'hex-g{}'.format(self.gid)
        eles = self.system.ele_map[etype]
        elesdata = eles._vect_upts
        ndim, nspt, neled1, nvars, neled2 = elesdata.datashape
        es =  nvars*neled2
        ss =  neled1*nvars*neled2 
        vs =  neled2
        ds =  neled1*nvars*neled2*nspt
        writeAt(ele_stride, 0, int(es))
        writeAt(spt_stride, 0, int(ss))
        writeAt(var_stride, 0, int(vs))
        writeAt(dim_stride, 0, int(ds))
        return int(elesdata.data)

    def get_nweights_gpu(self, cellid):
        eles, _ = self.find_eles_instance(cellid)
        return int(eles.basis.nupts)

    def _view(self, inter, meth, vshape=tuple()):
        # no permute
        self._perm = Ellipsis
        vm = _get_inter_objs(inter, meth, self.system.ele_map)
        vm = [np.concatenate(m)[self._perm] for m in zip(*vm)]
        return self.system.backend.view(*vm, vshape=vshape)
    
    def _scal_view_fpts_ploc(self, inter, meth):
        return self._view(inter, meth, (self.system.ndims,))

    def _scal_view_fpts_u(self, inter, meth):
        return self._view(inter, meth, (self.system.nvars, ))

    def _vect_view_fpts_du(self, inter, meth):
        return self._view(inter, meth, (self.system.ndims, self.system.nvars))