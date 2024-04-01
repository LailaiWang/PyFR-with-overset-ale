# -*- coding: utf-8 -*-
from opcode import hasname
from pickle import FALSE, TRUE
from mpi4py import MPI
import tioga as tg
from convert import *
import numpy as np
import time
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

    map_quad_2 = {0:[0,2],1:[2,3],2:[3,1],3:[1,0]}
    map_quad_3 = {0:[0,1,2],1:[2,5,8],2:[8,7,6],3:[6,3,0]}
    map_quad_4 = {0:[0,1,2,3],1:[3,7,11,15],2:[15,14,13,12],3:[12,8,4,0]}

    map_hex_2 = {0:[0, 2, 3, 1], 1:[0, 1, 5, 4], 2:[5,1,3,7],
                         3:[3, 2, 6, 7], 4:[0, 4,6,2], 5:[4, 5, 7, 6]}
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
        self.a=np.array([1, 2, 3, 4],dtype='int32')
    
    # int* cellid int* nnodes # of solution points basis.upts
    def get_nodes_per_cell(self, cellid, nnodes):
        cidx = ptrAt(cellid,0)
        etype = self.griddata['celltypes'][cidx]
        n = self.system.ele_map[etype].basis.nupts
        writeAt(nnodes,0,int(n))
        #exit()

    # int* faceid, int *nnodes # fpt
    ###needs optimization
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
        exit()
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
        exit()
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
        exit()
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
        exit()
        cid, nnspts = ptrAt(cellid,0), ptrAt(nspts,0)
        idout = cid*nnspts
        writeAt(index_out,0,idout)
        for i in range(nnspts):
            writeAt(q_out,i,ptrAt(q_in[i]))

    def donor_frac(self, cellid, xyz, nweights, inode, weights, rst, buffsize):
        exit()
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
    
    #needs optimization
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
        exit()
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
        exit()
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
        exit()
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
        exit()
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
        exit()
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
        exit()
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
    ### This is for testing the data layout of PyFR
    def check_fringe(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        if self.rank==0:
            ss='hex-g0'
    
            eles = self.system.ele_map[ss]
            elesdata = eles.scal_upts_inb._curr_mat
            Ploc=self.system.ele_map[ss].plocfpts
  
            for idx,m in enumerate(self.system._mpi_inters):
                AAA=m._scal_rhs.get()
                BBB=m._vect_rhs.get()
                name=f'foo-{idx}.dat'
                np.savetxt(name, BBB[1], delimiter=",")


    def fringe_data_to_device(self, fringeids, nfringe, gradflag, data, faceids, mpifringeid, nfringeface):
        if nfringe == 0: return

        if (self.system.tstep<2 and self.system.gridtype=='overset'):
            self.faceov_prep(fringeids,nfringe, faceids, mpifringeid, nfringeface)
        if (self.system.istage==0 and self.system.gridtype=='background'):
            self.faceov_prep(fringeids,nfringe, faceids, mpifringeid, nfringeface)

        if gradflag == 0: # passing u
            self.fringe_u_device(fringeids, nfringe, data)
        else: # passing du
            self.fringe_du_device(fringeids, nfringe, data)

    def faceov_prep(self, fringeids, nfringe,faceids,mpifringeid, nfringeface):
        # non-mpi fringe artbnd
        nvars=self.system.nvars
        # First deal with  inner artbnd
        tot_nfpts = 0
        faceinfo = []
        side_int = []
        faceinfo_test = []
        facedata = []
        facefringeid = []
        facenfpts = []
        fid_mpi=[]
        fringe_num=[]
        fid_mpi_num=[]
        mpi_fringe=0
        tot_nfpts_ov = 0
        faceinfo_ov = []
        facefpts_ov_range = [0]
        self.fid_mpi=0
        self.fringe_faceinfo = []
        self.tot_nfpts_fringe = 0
        self.nfringe=nfringe
        self.fid_mpi=0
        self.fpos_eidx={}

        fpos_fringe=[]
        cidx_fringe=[]
        facefringeid_ov = []
        facenfpts_ov = []
        
        if self.system.gridtype =='background':
            nfidmpi = nfringe - nfringeface
            for etype,ele in self.system.ele_map.items():
                if 'hex' in etype:
                    nfpts = self.system.ele_map[etype].basis.nfacefpts[0]
            self.nfpts = nfpts
            
            fid_mpi = []
            
            if nfidmpi > 0:
                pg = ptrToArray(mpifringeid,2*nfidmpi).reshape(2,-1) 
                for i in range(nfidmpi):
                    fid_mpi.append((int(pg[0][i]),int(pg[1][i])))
                
            self.fid_mpi=fid_mpi
              
            fff = ptrToArray(faceids,3*nfringeface).reshape(3,-1)

            self.facefringeid = fff[0]
            cidx_fringe = fff[1]
            fpos_fringe = fff[2]

            fpos_eidx = {'fpos':fpos_fringe,'cidx':cidx_fringe}

            if fpos_fringe.shape[0]>1:
                self.scal_fpts_u = self._scal_view_fpts_u_n(fpos_eidx, 'scal_fpts')
                if 'euler' not in str(self.system):
                    self.vect_fpts_du = self._vect_view_fpts_du_n(fpos_eidx, 'vect_fpts')
            self.fpos_eidx = fpos_eidx
            self.tot_nfpts_fringe = fff[0].shape[0]*nfpts
        
        if self.system.gridtype=='overset':
            # for overset grids

            for i in range(nfringe):
                fid = ptrAt(fringeids,i)
                cidx1, cidx2 = self.griddata['f2corg'][fid]
                fpos1, fpos2 = self.griddata['faceposition'][fid]
                etyp1 = self.griddata['celltypes'][cidx1]
                etyp2 = self.griddata['celltypes'][cidx2] 
                
                #print(f"rank is {self.rank} cidx1 is {cidx1} and cidx2 is {cidx2}")
                if cidx2 < 0 and cidx2 != -2:
                    #if cidx2 == -2: print(f'{cidx2} for {fid}')
                    # always use left info here
                    perface = (etyp1, cidx1 - self.griddata['celloffset'][cidx1], fpos1, 0) 
                    faceinfo_ov.append(perface)

                    facefringeid_ov.append(i)
                    # always use left face
                    nfpts = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
                    tot_nfpts_ov = tot_nfpts_ov + nfpts
                    facefpts_ov_range.append(tot_nfpts_ov)
                    # Grab the data
                    facenfpts_ov.append(nfpts)
                elif cidx2 > 0:
                    pass
                elif cidx < 0 and cidx2 == -2:
                    pass

            # save for later use
            self.fringe_faceinfo_ov = faceinfo_ov
            # save for later use
            self.tot_nfpts_ov = tot_nfpts_ov

            self.facefringeid_ov = facefringeid_ov
            self.facenfpts_ov = facenfpts_ov
            self.facefpts_ov_range=facefpts_ov_range
            
            if faceinfo_ov != []:

                nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts_ov
                nbytes = nbytes*self.system.nvars
                
                totnodesmap=[]
                temp_node=np.zeros((6,nfpts),dtype=int)
                for idx, face in enumerate(faceinfo_ov):
                    etype, typecidx, fpos, _ = face
                    srted_ord = self.system.ele_map[etype]._srtd_face_fpts[fpos][typecidx]
                    unsrted_ord = self.system.ele_map[etype].basis.facefpts[fpos]
                    # swap data according to the node ordering
                    nodesmap = []
                    for n in srted_ord:
                        nodesmap.append(unsrted_ord.index(n))
                    totnodesmap.append(nodesmap)
                    temp_node[fpos] = np.array(nodesmap)

                self.totnodesmap = np.array(totnodesmap)
                self.nodesmap = temp_node
                
    def fringe_u_stage(self):

        if self.faceinfo!=[] and self.system.gridtype=='background':
            
            tg.unpack_fringe_u_wrapper (
                addrToFloatPtr(self.fringe_u_fpts_d),
                addrToFloatPtr(int(self.scal_fpts_u._mats[0].basedata)),
                addrToUintPtr(self.scal_fpts_u.mapping.data),
                self.nfringe, self.tot_nfpts_fringe, self.system.nvars, self.backend.soasz, 3)
            #print('fringe_u_stage_done')

        if self.system.gridtype=='overset':
            
            tot_nfpts_ov =self.tot_nfpts_ov
            nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts_ov
            nbytes = nbytes*self.system.nvars
            tg.tg_copy_to_device(
                self.matrix_entry.data,
                addrToFloatPtr(self.cc.ctypes.data),
                int(nbytes)
            )
           
    def fringe_u_device(self, fringeids, nfringe, data):
        if nfringe == 0: return 0
        #print(sself.tot_nfpts_fringe)
        self.tot_nfpts_mpi=0
        #self.tot_nfpts_fringe=0
        nvars=self.system.nvars

        # First deal with  inner artbnd
        tot_nfpts = 0
        faceinfo = []
        side_int = []
        faceinfo_test = []
        
        facedata = []
        facefringeid = []
        facenfpts = []
        fid_mpi=[]
        fringe_num=[]
        fid_mpi_num=[]
        mpi_fringe=0
        p0=time.time()
        #self.fringe_faceinfo=[]
        if self.system.gridtype=='background':
            
            if self.facefringeid.shape[0]>0:

                self.fringe_u_fpts_d = self.griddata['fringe_u_fpts_d']
                faceinfo=self.fringe_faceinfo
                nfpts=self.nfpts
                tot_nfpts=self.tot_nfpts_fringe
                tot_nfpts_all=nfringe*nfpts

                facedata_tot=ptrToArray(data,tot_nfpts_all*nvars).reshape(nfringe,nfpts*nvars)
                facedata=facedata_tot[self.facefringeid,:].reshape(-1)
                
                self._scal_fpts_u=self.scal_fpts_u
                
                nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts
                nbytes = nbytes*nvars

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
                    nfringe, tot_nfpts, nvars, self.backend.soasz, 3
                )

        if self.system.gridtype=='background':
            #self.tioga_pass_data(arrayToIntPtr(self.a))
            for midx, mpiinters in enumerate(self.system._mpi_inters):
                # then deal with overset artbnd
                scal_size= len(mpiinters.lhs)
                tot_nfpts_mpi = 0
                faceinfo_mpi = []
                facefpts_mpi_range = [0]
                facefringeid_mpi = []
                facedata_mpi = []
                facenfpts_mpi = []

                for idx_fid, (i,fid) in enumerate(self.fid_mpi):
                    #fid = ptrAt(fringeids,i)
                    cidx1, cidx2 = self.griddata['f2corg'][fid]
                    fpos1, fpos2 = self.griddata['faceposition'][fid]
                    etyp1 = self.griddata['celltypes'][cidx1]
                    etyp2 = self.griddata['celltypes'][cidx2] 
                    fid_mpi_list=[]
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
                        fid_mpi_list.append(fid)
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

                        a = facedata_mpi[facefpts_mpi_range[idx]:facefpts_mpi_range[idx+1]]

                        startIdx = mpiinters.lhs.index(face)

                        dataperface = a[nodesmap]
                        
                        cc = dataperface.swapaxes(0,1).reshape(-1)

                        matrix_entry = mpiinters._scal_rhs
                        
                        # Need to figure out the locations of these faces in the matrix entry
                        nfpts = facenfpts_mpi[idx]
                        nvars=self.system.nvars
                        for i in range(nvars):
                            cc_var=cc[i*nfpts:(i+1)*nfpts]
                            nbytes = np.dtype(self.backend.fpdtype).itemsize*nfpts
                            #nbytes = nbytes*self.system.nvars
                            # calculate the offset of this face in the matrix entry
                            offset = startIdx*nbytes+i*scal_size*nfpts*np.dtype(self.backend.fpdtype).itemsize
                            #tg.tg_copy_to_device(matrix_entry.data,data,int(nbytes))
                            tg.tg_copy_to_device(
                                matrix_entry.data+offset,
                                addrToFloatPtr(cc_var.ctypes.data),
                                int(nbytes)
                            )                        
                            
                        #exit()
        # Then deal with overset artbnd
        if self.system.gridtype=='overset':

            tot_nfpts_ov =self.tot_nfpts_ov
            faceinfo_ov=self.fringe_faceinfo_ov 
            facefpts_ov_range = self.facefpts_ov_range
            nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts_ov
            nbytes = nbytes*self.system.nvars
            facedata_ov=ptrToArray(data,tot_nfpts_ov*5)
            facedata_ov = facedata_ov.reshape(-1,5)
            for idx, face in enumerate(faceinfo_ov):
                etype, typecidx, fpos, _ = face
                #nodesmap=self.totnodesmap[idx]
                nodesmap=self.nodesmap[fpos]
                # swap data 
                a = facedata_ov[facefpts_ov_range[idx]:facefpts_ov_range[idx+1]]
                facedata_ov[facefpts_ov_range[idx]:facefpts_ov_range[idx+1]] = a[nodesmap]

            cc = facedata_ov.swapaxes(0,1).reshape(-1)
            
            matrix_entry = self.system._mpi_inters[-1]._scal_rhs
            self.matrix_entry=matrix_entry
            self.cc=cc
            
            tg.tg_copy_to_device(
                matrix_entry.data,
                addrToFloatPtr(cc.ctypes.data),
                int(nbytes)
            )

    def fringe_du_device(self, fringeids, nfringe, data):
        
        if self.tot_nfpts_fringe + self.tot_nfpts_mpi==0:
            return 0
        if self.tot_nfpts_fringe>0:
            tot_nfpts=self.tot_nfpts_fringe
            # Deal with interior artificial boundary conditions
            #faceinfo = self.fringe_faceinfo
            # copy data to device
            self.fringe_du_fpts_d = self.griddata['fringe_du_fpts_d']
            nbytes = np.dtype(self.backend.fpdtype).itemsize*tot_nfpts
            nbytes = nbytes*self.system.nvars*self.system.ndims
            nvars =self.system.nvars
            tot_nfpts_all=nfringe*self.nfpts
            facedata_tot=ptrToArray(data,tot_nfpts_all*nvars*self.system.ndims).reshape(
                nfringe,self.system.ndims*nvars*self.nfpts)
            facedata=facedata_tot[self.facefringeid,:].reshape(-1)

            facedata = np.array(facedata).astype(self.backend.fpdtype)

            tg.tg_copy_to_device(
                self.fringe_du_fpts_d, 
                addrToFloatPtr(facedata.__array_interface__['data'][0]),
                int(nbytes)
            )

            self._vect_fpts_du =  self.vect_fpts_du     
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

    def get_face_nodes_gpu(self, faceids, nfaces, nptsface, xyz, fdata):
        # these faces are
        if nfaces == 0: return
        fpos_fringe=[]
        cidx_fringe=[]
        faceinfo = []    
        tot_nfpts = 0
        tot_nfpts=int(np.sum(ptrToArray(nptsface,nfaces)))
        ff=ptrToArray(fdata,nfaces*2).reshape(2,-1)
        fpos_fringe=ff[0]
        cidx_fringe=ff[1]
        #fpos_fringe = np.array(fpos_fringe)
        #cidx_fringe =  np.array(cidx_fringe)
        fpos_eidx = {'fpos':fpos_fringe,'cidx':cidx_fringe}
        # note there need to use unsorted face idx
        # check base/element.py

        #self._scal_fpts_ploc = self._scal_view_fpts_ploc(
        #    faceinfo, 'get_scal_unsrted_fpts_ploc_for_inter')
        
        ########new implementation by Amir
        self._scal_fpts_ploc = self._scal_view_fpts_ploc_n( fpos_eidx,'ploc')
        #    exit()
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
        p0=time.time()
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
        p1=time.time()
        #print('get cell nodes gpu',p1-p0)
        
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

    
    ### Should be developed in TIOGA for optimization purpose 

    def _get_mid(self,etype,meth):

        if 'scal_fpts' in meth:
            return (self.system.ele_map[etype]._scal_fpts.mid)
        elif 'ploc' in meth:
            return (self.system.ele_map[etype].ploc_at_ncon('fpts').mid)
        elif 'vect' in meth:    
            return (self.system.ele_map[etype]._vect_fpts.mid)
        else: 
             raise RuntimeError('get_mid not right')

    
    def _calc_view(self, inter,meth): 
        for etype,ele in self.system.ele_map.items():

            if 'hex' not in etype:
                raise RuntimeError('wrong element type: Only HEX is approved')
            else:
                mid=self._get_mid(etype,meth)
                nfp = self.system.ele_map[etype].basis.nfacefpts[0]
                fpos=inter['fpos']
                cidx=inter['cidx']    
                cidx_arr=np.repeat(cidx,repeats=nfp, axis=0)
                mid_arr=np.full((cidx_arr.shape[0]),mid)
                facefpts=np.array(self.system.ele_map[etype].basis.facefpts)#, dtype='int32')
                #ff=gg.reshape(-1)
                
                lfunc = lambda i:facefpts[i]
                fpos_arr = lfunc(fpos).reshape(-1)
                #print(fpos,cidx)
                vm=[]
                vm=[mid_arr,fpos_arr,cidx_arr]
                if meth == 'vect_fpts':
                    rstri_arr=np.full((cidx_arr.shape[0]),self.system.ele_map[etype].nfpts)
                    vm.append(rstri_arr)
                return(vm)

    def _view_n(self, inter, meth, vshape=tuple()):
        # no permute
        vm=self._calc_view(inter, meth)
        return self.system.backend.view(*vm, vshape=vshape)
    

    def _scal_view_fpts_u_n(self, inter,meth):
        return self._view_n(inter, meth, (self.system.nvars, ))

    def _vect_view_fpts_du_n(self, inter,meth):
        return self._view_n(inter, meth, (self.system.ndims, self.system.nvars))

    def _scal_view_fpts_ploc_n(self, inter, meth):
        return self._view_n(inter, meth, (self.system.ndims,))

