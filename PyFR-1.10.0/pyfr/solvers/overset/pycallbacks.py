# -*- coding: utf-8 -*-
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
        self.setup_interior_mapping()
        self.setup_mpi_mapping()
        self.setup_mpi_rhs_mapping()
        self.setup_bc_rhs_basedata()
        self.setup_bc_mapping()
        self.setup_structured_to_srted_mapping()
        self.setup_facecoords_mapping()
        self.setup_cell_info_by_type()
        # set up mpi artbnd status
        self.setup_mpi_artbnd_status_aux()

    def setup_structured_to_srted_mapping(self):
        grid = self.griddata
        ncells, ctype, off = grid['ncells'], grid['celltypes'], grid['celloffset']
        maxnfpts, maxnface = grid['maxnfpts'], grid['maxnface']
        
        ele_map = self.system.ele_map
        
        rf = lambda cidx : cidx - off[cidx]
        get_ctype = lambda cidx: ctype[cidx]
        get_nface  = lambda ctype : len(ele_map[ctype].basis.faces)
        
        # add necessary padding to these two
        get_unsrted = (lambda ctype: [ ele_map[ctype].basis.facefpts[fpos] +
              (maxnfpts - len(ele_map[ctype].basis.facefpts[fpos])) * [0] 
              if fpos < get_nface(ctype) else maxnfpts * [0] for fpos in range(maxnface)
            ])
        get_srted = (lambda ctype,cidx: [list(ele_map[ctype]._srtd_face_fpts[fpos][rf(cidx)]) + 
              (maxnfpts - len(list(ele_map[ctype]._srtd_face_fpts[fpos][rf(cidx)]))) * [0]
              if fpos < get_nface(ctype) else maxnfpts * [0] for fpos in range(maxnface)
            ])
        
        unsrted_info = [ get_unsrted(get_ctype(i)) for i in range(ncells)]
        srted_info = [get_srted(get_ctype(i), i) for i in range(ncells)]
        unsrted_info = np.array(unsrted_info).astype('int32').reshape(-1)
        srted_info = np.array(srted_info).astype('int32').reshape(-1)
        
        tg.tioga_set_data_reorder_map(srted_info.ctypes.data, unsrted_info.ctypes.data, ncells)

    def setup_bc_rhs_basedata(self):
        # set this for overset grid
        mpi_inters = self.system._mpi_inters
        if mpi_inters != [] and mpi_inters[-1]._rhsrank == None:
            basedata = mpi_inters[-1]._scal_rhs.data
            tg.tioga_set_bc_rhs_basedata(basedata)
            

    def setup_bc_mapping(self):
        grid = self.griddata
        nfaces, nbcfaces = grid['nfaces'], grid['nbcfaces']
        if nbcfaces == 0: return    
    
        f2c, fpos = grid['f2corg'], grid['faceposition'] 
        ctype, off = grid['celltypes'], grid['celloffset']
        
        ctype_to_int_dict = {'hex': 8}
        ctype_to_int = lambda ctype : ctype_to_int_dict[ctype.split('-')[0]]
        info_in_int = lambda fid: [ctype_to_int(ctype[f2c[fid][0]]),f2c[fid][0],fpos[fid][0]]
        rf = lambda cidx : cidx - off[cidx]
        info_in_tuple = lambda fid: (ctype[f2c[fid][0]],rf(f2c[fid][0]),fpos[fid][0],0)
        
        bc_side_in_int = [info_in_int(fid) for fid in range(nbcfaces)]
        bc_side_in_tuple = [info_in_tuple(fid) for fid in range(nbcfaces)]

        bc_fpts_artbnd = self._scal_view_artbnd(
            bc_side_in_tuple, 'get_scal_fpts_artbnd_for_inter'
        )

        get_nfpts = lambda etype, pos : self.system.ele_map[etype].basis.nfacefpts[pos]
        bc_side_nfpts = [get_nfpts(etype, pos) for etype,_,pos,_ in bc_side_in_tuple]

        pad = lambda a,n: np.concatenate((np.tile(a,(n,1)),np.array(range(n)).reshape(-1,1)),axis=1)

        bc_side_in_int  = [ pad(a,n) for a, n in zip(bc_side_in_int, bc_side_nfpts)]
        bc_side_in_int = np.array(bc_side_in_int).reshape(-1,4).reshape(-1).astype('int32')
        
        bc_mapping = bc_fpts_artbnd.mapping.get().squeeze(0)

        basedata = int(bc_fpts_artbnd.basedata)
        tg.tioga_set_bc_mapping(basedata, bc_side_in_int.ctypes.data, bc_mapping.ctypes.data, bc_mapping.shape[0])

    def setup_mpi_rhs_mapping(self):
        mpi_inters = self.system._mpi_inters
        if mpi_inters == []: return
        
        grid = self.griddata
        rhs_basedata = self.system._mpi_inters[0]._scal_rhs.data
        face_mapping = grid['mpientry'] # entry for each face
        mpi_nfpts = grid['mnfpts'] # fpts per mpi face
        face_strides = grid['mbaseface'] # stride for each variable
        nmpifaces = grid['nmpifaces']
        
        itemsize = np.dtype(self.system.backend.fpdtype).itemsize
        # note that facemapping is in bytes
        # now we populate it for each flux points
        fpts_rhs_mapping = [ list(face_mapping[i] + itemsize*np.arange(mpi_nfpts[i])) for i in range(nmpifaces)]
        fpts_rhs_strides = [ [face_strides[i]] * mpi_nfpts[i] for i in range(nmpifaces)] 
        fpts_rhs_mapping = np.array(fpts_rhs_mapping).astype('int32').reshape(-1)
        fpts_rhs_strides = np.array(fpts_rhs_strides).astype('int32').reshape(-1)
        
        tg.tioga_set_mpi_rhs_mapping(rhs_basedata, fpts_rhs_mapping.ctypes.data, fpts_rhs_strides.ctypes.data, fpts_rhs_mapping.shape[0])        

    def setup_mpi_mapping(self):
        grid = self.griddata
        nfaces, nbcfaces, nmpifaces = grid['nfaces'], grid['nbcfaces'], grid['nmpifaces']
        if nmpifaces == 0: return    
    
        f2c, fpos = grid['f2corg'], grid['faceposition'] 
        ctype, off = grid['celltypes'], grid['celloffset']
        
        ctype_to_int_dict = {'hex': 8}
        ctype_to_int = lambda ctype : ctype_to_int_dict[ctype.split('-')[0]]
        info_in_int = lambda fid: [ctype_to_int(ctype[f2c[fid][0]]),f2c[fid][0],fpos[fid][0]]
        rf = lambda cidx : cidx - off[cidx]
        info_in_tuple = lambda fid: (ctype[f2c[fid][0]],rf(f2c[fid][0]),fpos[fid][0],0)
        
        mpi_side_in_int = [info_in_int(fid) for fid in range(nbcfaces, nbcfaces+nmpifaces)]
        mpi_side_in_tuple = [info_in_tuple(fid) for fid in range(nbcfaces, nbcfaces+nmpifaces)]

        mpi_fpts_artbnd = self._scal_view_artbnd(
            mpi_side_in_tuple, 'get_scal_fpts_artbnd_for_inter'
        )

        get_nfpts = lambda etype, pos : self.system.ele_map[etype].basis.nfacefpts[pos]
        mpi_side_nfpts = [get_nfpts(etype, pos) for etype,_,pos,_ in mpi_side_in_tuple]

        pad = lambda a,n: np.concatenate((np.tile(a,(n,1)),np.array(range(n)).reshape(-1,1)),axis=1)

        mpi_side_in_int  = [ pad(a,n) for a, n in zip(mpi_side_in_int, mpi_side_nfpts)]
        mpi_side_in_int = np.array(mpi_side_in_int).reshape(-1,4).reshape(-1).astype('int32')
        
        mpi_mapping = mpi_fpts_artbnd.mapping.get().squeeze(0)

        basedata = int(mpi_fpts_artbnd.basedata)
        tg.tioga_set_mpi_mapping(basedata, mpi_side_in_int.ctypes.data, mpi_mapping.ctypes.data, mpi_mapping.shape[0])
     
    def setup_facecoords_mapping(self):   
        grid = self.griddata
        nfaces, nbcfaces, nmpifaces = grid['nfaces'], grid['nbcfaces'], grid['nmpifaces']
        f2c, fpos = grid['f2corg'], grid['faceposition']
        ctype, off = grid['celltypes'], grid['celloffset']
        
        # now only support this for hex
        ctype_to_int_dict = {'hex': 8}
        # define some lambdas
        ctype_to_int = lambda ctype : ctype_to_int_dict[ctype.split('-')[0]]
        info_in_int = lambda fid, i: [ctype_to_int(ctype[f2c[fid][i]]),f2c[fid][i],fpos[fid][i]]
        rf = lambda cidx : cidx - off[cidx] # cell idx in its own type is needed for tuple
        info_in_tuple = lambda fid, i: (ctype[f2c[fid][i]],rf(f2c[fid][i]),fpos[fid][i],0)
        
        # get left and right separately
        left_side_in_int = [info_in_int(fid, 0) for fid in range(nfaces)]
        left_side_in_tuple = [info_in_tuple(fid, 0) for fid in range(nfaces)]
        
        # new generate the information
        fpts_ploc_left = self._scal_view_fpts_ploc(
            left_side_in_tuple, 'get_scal_unsrted_fpts_ploc_for_inter'
        )

        get_nfpts = lambda etype, pos : self.system.ele_map[etype].basis.nfacefpts[pos]
        left_side_nfpts = [get_nfpts(etype, pos) for etype,_,pos,_ in left_side_in_tuple]

        pad = lambda a,n: np.concatenate((np.tile(a,(n,1)),np.array(range(n)).reshape(-1,1)),axis=1)
        left_side_in_int  = [ pad(a,n) for a, n in zip(left_side_in_int, left_side_nfpts)]
        left_side_in_int = np.array(left_side_in_int).reshape(-1,4).reshape(-1).astype('int32')
        left_mapping = fpts_ploc_left.mapping.get().squeeze(0).astype('int32')
        basedata = int(fpts_ploc_left.basedata)
        tg.tioga_set_facecoords_mapping(
            basedata, left_side_in_int.ctypes.data, left_mapping.ctypes.data, left_mapping.shape[0]
        )
    
    def setup_cell_info_by_type(self):
        grid = self.griddata
        ele_map = self.system.ele_map
        ctype, off  = grid['celltypes'], grid['celloffset']
        ntypes, ncells = grid['ncelltypes'], grid['ncells']
        # now only support this for hex
        ctype_to_int_dict = {'hex': 8}
        # define some lambdas
        ctype_to_int = lambda ctype : ctype_to_int_dict[ctype.split('-')[0]]
        celltypes  = np.array([ ctype_to_int(a) for a in ctype]).astype('int32')
        nupts_per_type = np.array( 
                [[ctype_to_int(etype), eles.basis.nupts] for etype, eles in ele_map.items()]
            ).astype('int32').reshape(-1)
        
        get_strides = lambda s: [s[2]*s[3],s[1]*s[2]*s[3],s[3]]
        get_datashape = lambda eles : eles.scal_upts_inb._curr_mat.datashape
        # now get strides for state variable
        ustrides = np.array(
                [[ctype_to_int(etype)] + get_strides(get_datashape(eles)) for etype, eles in ele_map.items() ]
            ).astype('int32').reshape(-1)
        get_du_strides = lambda s: [s[3]*s[4], s[2]*s[3]*s[4], s[4], s[1]*s[2]*s[3]*s[4]]
        get_du_datashape = lambda eles: eles._vect_upts.datashape
        dustrides = np.array(
                [[ctype_to_int(etype)] + get_du_strides(get_du_datashape(eles)) for etype, eles in ele_map.items() ]
            ).astype('int32').reshape(-1)
        # memory address for u is subject to change at different stages of RK methods,
        # we need to get it dynamically
        # this is memory address use int64
        du_basedata = np.array([
            [ctype_to_int(etype),int(eles._vect_upts.data)] for etype, eles in ele_map.items()]
            ).astype('int64').reshape(-1)

        cstrides = np.array([

            [ctype_to_int(etype)] + eles.ploc_at_ncon('upts').datashape for etype, eles in ele_map.items()
            ]).astype('int32').reshape(-1)
        
        coords_basedata = np.array([
            [ctype_to_int(etype),int(eles.ploc_at_ncon('upts').data)] for etype, eles in ele_map.items()
            ]). astype('int64').reshape(-1)
            
        # now get strides for gradient variables
        tg.tioga_set_cell_info_by_type(
                ntypes, ncells,
                celltypes.ctypes.data, 
                nupts_per_type.ctypes.data, 
                ustrides.ctypes.data, dustrides.ctypes.data, 
                du_basedata.ctypes.data, 
                cstrides.ctypes.data, coords_basedata.ctypes.data
            )
                   
    def setup_interior_mapping(self):
        grid = self.griddata
        nfaces, nbcfaces, nmpifaces = grid['nfaces'], grid['nbcfaces'], grid['nmpifaces']
        f2c, fpos = grid['f2corg'], grid['faceposition']
        ctype, off = grid['celltypes'], grid['celloffset']
        
        # now only support this for hex
        ctype_to_int_dict = {'hex': 8}
        # define some lambdas
        ctype_to_int = lambda ctype : ctype_to_int_dict[ctype.split('-')[0]]
        info_in_int = lambda fid, i: [ctype_to_int(ctype[f2c[fid][i]]),f2c[fid][i],fpos[fid][i]]
        rf = lambda cidx : cidx - off[cidx] # cell idx in its own type is needed for tuple
        info_in_tuple = lambda fid, i: (ctype[f2c[fid][i]],rf(f2c[fid][i]),fpos[fid][i],0)
        
        # get left and right separately
        left_side_in_int = [info_in_int(fid, 0) for fid in range(nbcfaces+nmpifaces, nfaces)]
        left_side_in_tuple = [info_in_tuple(fid, 0) for fid in range(nbcfaces+nmpifaces, nfaces)]
        
        # new generate the information
        fpts_u_left = self._scal_view_fpts_u(
            left_side_in_tuple, 'get_scal_unsrted_fpts_for_inter'
        )

        fpts_du_left = self._vect_view_fpts_du(
            left_side_in_tuple, 'get_vect_unsrted_fpts_for_inter'
        )

        get_nfpts = lambda etype, pos : self.system.ele_map[etype].basis.nfacefpts[pos]
        left_side_nfpts = [get_nfpts(etype, pos) for etype,_,pos,_ in left_side_in_tuple]

        pad = lambda a,n: np.concatenate((np.tile(a,(n,1)),np.array(range(n)).reshape(-1,1)),axis=1)
        left_side_in_int  = [ pad(a,n) for a, n in zip(left_side_in_int, left_side_nfpts)]
        left_side_in_int = np.array(left_side_in_int).reshape(-1,4).reshape(-1).astype('int32')
        left_mapping = fpts_u_left.mapping.get().squeeze(0).astype('int32')
        left_grad_mapping = fpts_du_left.mapping.get().squeeze(0).astype('int32')
        left_grad_strides = fpts_du_left.rstrides.get().squeeze(0).astype('int32')
        
        # now lets do the same thing for right side
        righ_side_in_int = [info_in_int(fid, 1) for fid in range(nbcfaces+nmpifaces, nfaces)]
        righ_side_in_tuple = [info_in_tuple(fid, 1) for fid in range(nbcfaces+nmpifaces, nfaces)]
        # new generate the information
        fpts_u_righ = self._scal_view_fpts_u(
            righ_side_in_tuple, 'get_scal_unsrted_fpts_for_inter'
        )
      
        fpts_du_righ = self._vect_view_fpts_du(
            righ_side_in_tuple, 'get_vect_unsrted_fpts_for_inter'
        )
        
        righ_side_nfpts = [get_nfpts(etype, pos) for etype,_,pos,_ in righ_side_in_tuple]
        righ_side_in_int  = [ pad(a,n) for a, n in zip(righ_side_in_int, righ_side_nfpts)]
        righ_side_in_int = np.array(righ_side_in_int).reshape(-1,4).reshape(-1).astype('int32')
        righ_mapping = fpts_u_righ.mapping.get().squeeze(0).astype('int32')
        righ_grad_mapping = fpts_du_righ.mapping.get().squeeze(0).astype('int32')
        righ_grad_strides = fpts_du_righ.rstrides.get().squeeze(0).astype('int32')

        sum_info = np.concatenate((left_side_in_int, righ_side_in_int)).astype('int32')
        # this could exceed int
        sum_mapping = np.concatenate((left_mapping, righ_mapping)).astype('int32')
        # now we set the information on tioga
        sum_grad_mapping = np.concatenate((left_grad_mapping, righ_grad_mapping)).astype('int32')        
        sum_grad_strides = np.concatenate((left_grad_strides, righ_grad_strides)).astype('int32')

        if int(fpts_u_left.basedata) != int(fpts_u_righ.basedata):
            raise RuntimeError("Interior left and right basedata not consistent")
        # need to copy this base data as well
        basedata = int(fpts_u_left.basedata)
        if int(fpts_du_left.basedata) != int(fpts_du_righ.basedata):
            raise RuntimeError("Interior left and right grad basedata not consistent")
        grad_basedata = int(fpts_du_left.basedata)
        tg.tioga_set_interior_mapping(
            basedata, grad_basedata, sum_info.ctypes.data, sum_mapping.ctypes.data, 
            sum_grad_mapping.ctypes.data, sum_grad_strides.ctypes.data, sum_mapping.shape[0]
        )

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
        etyp2  = self.griddata['celltypes'][cidx2] if cidx2 != -1 else etyp1
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

    def setup_mpi_artbnd_status_aux(self):
        # information of all mpi interfaces
        nbcfaces = self.griddata['nbcfaces']
        nmpifaces = self.griddata['nmpifaces']
        if nmpifaces == 0:
            self._mpi_fpts_artbnd = None
            return
 
        tot_nfpts_mpi = 0
        faceinfo_mpi = []
        facefpts_mpi_range = []
        for fid in range(nbcfaces, nbcfaces + nmpifaces):
            cidx1, cidx2 = self.griddata['f2corg'][fid]
            fpos1, _ = self.griddata['faceposition'][fid]
            etyp1 = self.griddata['celltypes'][cidx1]
            etyp2 = self.griddata['celltypes'][cidx2]
            if cidx2 >= 0:
                raise RuntimeError(f'Right cell of a mpi face shoud not be {cidx2}')
            
            perface = (etyp1, cidx1 - self. griddata['celloffset'][cidx1], fpos1, 0)
            faceinfo_mpi.append(perface)

            nfpts = self.system.ele_map[etyp1].basis.nfacefpts[fpos1]
            tot_nfpts_mpi = tot_nfpts_mpi + nfpts
            facefpts_mpi_range.append(tot_nfpts_mpi)

        self.init_info_mpi = faceinfo_mpi
        self.init_nfpts_mpi = tot_nfpts_mpi
        self.init_mpi_range = facefpts_mpi_range

        self._init_fpts_artbnd = self._scal_view_artbnd(
            self.init_info_mpi, 'get_scal_fpts_artbnd_for_inter'
        )
        
        # tioga function to reset the status value to -1
        tg.reset_mpi_face_artbnd_status_wrapper(
            addrToFloatPtr(int(self._init_fpts_artbnd._mats[0].basedata)),
            addrToIntPtr(self._init_fpts_artbnd.mapping.data),
            -1.0,
            nmpifaces, self.init_nfpts_mpi, 1, self.backend.soasz, 3
        )

    def fringe_data_to_device(self, fringeids, nfringe, gradflag, data):
        if nfringe == 0: return
        if self.system.istage == 0: # we only need to find the information for first stage
            tg.tioga_update_fringe_face_info(gradflag)
            tg.tioga_reset_entire_mpi_face_artbnd_status_pointwise(1)
            tg.tioga_reset_mpi_face_artbnd_status_pointwise(1)
        
        if gradflag == 0: # passing u
            self.fringe_u_device(fringeids, nfringe, data)
        else: # passing du
            self.fringe_du_device(fringeids, nfringe, data)
    
    def fringe_u_device(self, fringeids, nfringe, data):
        tg.tioga_prepare_mpi_artbnd_target_data(data, self.system.nvars)            
        tg.tioga_prepare_interior_artbnd_target_data(data, self.system.nvars)
        tg.tioga_prepare_overset_artbnd_target_data(data, self.system.nvars)       
 
    def fringe_du_device(self, fringeids, nfringe, data):
        tg.tioga_prepare_interior_artbnd_target_data_gradient(
            data, self.system.nvars, self.system.ndims
        )

    def get_q_spts_gpu(self, et):
        int_to_ctype_dict = {8: 'hex'}
        etype = f'{int_to_ctype_dict[et]}-g{self.gid}'
        return int(self.system.ele_map[etype].scal_upts_inb._curr_mat.data)

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

    def _scal_view_artbnd(self, inter, meth):
        return self._view(inter, meth, (1,))
