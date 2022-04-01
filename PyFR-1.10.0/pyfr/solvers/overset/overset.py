# -*- coding: utf-8 -*-
from convert import *
import tioga as tg

import numpy as np
from mpi4py import MPI
from collections import defaultdict
from pyfr.nputil import fuzzysort
from collections import defaultdict, OrderedDict
from pyfr.solvers.overset.pycallbacks import Py_callbacks, face_vidx_incell, structured_hex_to_gmsh

class Overset(object):
    def __init__(self, system, rallocs):
        
        self.nGrids = system.ngrids
        self.gid = system.gid
        self.system = system
        
        # data type for tioga
        # right now integer must be int32 floats must be float64
        self.intdtype = 'int32'
        self.fpdtype = self.system.backend.fpdtype
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nproc = comm.Get_size()
        
        # wrap up each grids into one communicator
        self.gridcomm = comm.Split(self.gid,rank)
        self.gridrank = self.gridcomm.Get_rank()
        self.gridsize = self.gridcomm.Get_size()
        
        llrank= self.gridrank 
        lrank=np.array(llrank,dtype=np.int)
        grank=np.zeros((nproc), dtype=np.int)
        comm.Allgather([lrank, MPI.INT], [grank, MPI.INT])
        self.grrank=grank

        # initialize the MPI communicator
        tg.tioga_init_(comm)
    
        # initialize some basic information
        self.ndims, self.nvars = self.system.ndims, self.system.nvars
        self.nfields = system.nvars
        self.motion = system.mvgrid
        self.gridScale = 1.0
        
        # determine if we need GPU tioga
        if 'CUDA' not in type(system.backend).__name__:
            self.useGpu = False
        else:
            self.useGpu = True
            tg.initialize_stream_event()
            self.cudastream = tg.get_stream_handle()
            self.cudaevent = tg.get_event_handle()

        self.facevidxcell = face_vidx_incell()
        
        self.griddata = self._prepare_griddata()
        self.callbacks = Py_callbacks(self.system, self.griddata)
        self._init_overset()
        self._preprocess()
        self._performConnectivity()

        if self.system.restart is True:
            self.process_restart()
            self._performConnectivity()
        
        self.name = 'tioga'

    
    def _form_unique_nodes(self, tol = 1e-9):
        '''
        This subroutine finds the unique nodes in the mesh
        '''
        elemap = self.system.ele_map
        
        coords = []  # coordinates
        nnodes_offset = [0] # offset for points

        nspt_etypes = [] # number of pnts  for each cell type
        nfcs_etypes = [] # number of faces for each cell type
        nele_etypes = [] # number of cells for each cell type

        elesbasis = defaultdict()
         
        for idx, (etype, eles) in enumerate(elemap.items()):
            xyz = eles.eles.swapaxes(0, 1).reshape(-1, self.ndims)
            nnodes_offset.append(xyz.shape[0] + nnodes_offset[idx])
            coords.append(xyz)

            nspt_etypes.append(eles.eles.shape[0])
            nfcs_etypes.append(len(eles.basis.faces))
            nele_etypes.append(eles.eles.shape[1])

            elesbasis[etype.split('-g')[0]] = eles.basis

        maxnspt, maxnfcs = max(nspt_etypes), max(nfcs_etypes)
        # keep track of some small things
        self.elesbasis, self.maxnspt, self.maxnfcs = elesbasis, maxnspt, maxnfcs

        # dtype is crucial important for swig
        nspt_etypes = np.array(nspt_etypes).astype(self.intdtype)
        nfcs_etypes = np.array(nfcs_etypes).astype(self.intdtype)
        nele_etypes = np.array(nele_etypes).astype(self.intdtype)

        # keep track of small things
        self.nspt_etypes = nspt_etypes
        self.nfcs_etypes = nfcs_etypes
        self.nele_etypes = nele_etypes

        # merge nodes
        coords = np.concatenate(coords, axis = 0)
        
        srtd = fuzzysort(coords.T.tolist(), range(coords.shape[0]))

        # coords = coords[srtd]
        # generate unique indices for nodes
        nodesmap = [0]*coords.shape[0]
        nodesidx = 0
        pts = coords[srtd[0]]
        for idxsrtd in srtd[1:]:
            # find new nods
            x = coords[idxsrtd]
            # compare with designated tolerance
            if np.linalg.norm(x-pts)>tol:
                pts = x
                nodesidx = nodesidx+1
            nodesmap[idxsrtd] = nodesidx

        nodesmap = np.array(nodesmap).astype(self.intdtype)

        # get unique coordinates placed in order
        unikcoords = np.zeros((nodesidx+1, coords.shape[1]))
        for idx, i in enumerate(nodesmap):
            unikcoords[i] = coords[idx]
        

        nnodes = unikcoords.shape[0]
        self.nnodes = nnodes

        # flattern the coordinates
        unikcoords = unikcoords.reshape(-1).astype(self.fpdtype)
        return unikcoords, nnodes_offset, nodesmap

    def _form_cell_info(self, nnodes_offset, nodesmap):
        '''
        form cell connectivity information
        figure out c2v, celltype, initialize c2f
        '''
        elemap = self.system.ele_map
        c2v       = [ ]
        c2f       = [ ]
        celltypes = [ ] 
        nc2v      = [ ]
        coffset   = [0]
        for idx, (etype, eles) in enumerate(elemap.items()):
            npts, neles = eles.eles.shape[0], eles.eles.shape[1]
            b = np.tile(list(range(npts)), (neles, 1))
            b += (np.arange(neles)*npts)[:,None]
            # add offset think about this again
            b = b + nnodes_offset[idx]
            b = np.apply_along_axis(lambda x: nodesmap[x], 1, b)
            # pad -1 to axis to make sure dimension consistency 
            if self.nspt_etypes[idx] < self.maxnspt :
                pad = np.tile([-1]*(self.maxnspt-self.nspt_etypes[idx]),(neles,1))
                b = np.concatenate((b,pad),axis = 1)

            c2v.append(b)
            # initialize c2f
            fcs = np.tile([-1]*self.maxnfcs, (neles,1)) 
            c2f.append(fcs)

            ctype = np.array([etype]*neles)
            celltypes.append(ctype)

            coffset.append(neles)

        celloffset = []
        for idx, start in enumerate(coffset[:-1]):
            neles = coffset[idx+1]
            celloffset = celloffset + [sum(coffset[:idx+1])]*neles

        celloffset = np.array(celloffset).astype(self.intdtype)

        c2v = np.concatenate(c2v, axis = 0)
        c2f = np.concatenate(c2f, axis = 0)
        celltypes = np.concatenate(celltypes, axis = 0)
        
        # save number of cells
        self.ncells = c2v.shape[0]
        # return some intermediate variables
        return c2v, c2f, celltypes, celloffset

    def _form_face_info(self, c2v, c2f):
        '''
        Form the cell to face information   
        In PyFR, a face could be:
            (1) interior face
            (2) bc face
            (3) mpi face
        In this subroutine, we deal with bc face first.
        Overset face is technically created as mpi_inters object in PyFR
        with rhsrank as None

        One special face is the interior face which was periodic faces.
        '''
        self.int_periodic = 0 # this one is causing problem

        elemap = self.system.ele_map
        elesbasis = self.elesbasis
        
        # two useful/heavily-used lambda expressions
        fetype = lambda etype, fidx: elesbasis[etype].faces[fidx][0]
        fnodes = (
            lambda etype, eidx, fidx:
            c2v[eidx][self.facevidxcell[etype+'_{}'.format(elesbasis[etype].nsptsord)][fidx]] 
        ) 
        
        # lists of interfaces
        # most of things in PYTHON are pointers
        int_inters = self.system._int_inters
        bc_inters  = self.system._bc_inters
        mpi_inters = self.system._mpi_inters
        
        # Treat overset mpiinters as bc here
        # need to move overset inters from mpi_inters into bc_inters
        for mpiint in mpi_inters:
            if mpiint._rhsrank == None:
                bc_inters.append(mpiint)
                mpi_inters.remove(mpiint)
        
        # bc interfaces
        bfacetypes =  []
        bf2v = []
        bf2c = []
        bfposition = []
        binstanceid = []
        bcoffset = [0]

        # For parallel computing partitions could possibly have no bc inters
        if bc_inters != []:
            for idx, a in enumerate(bc_inters):
                bc_facetypes = np.array(
                    [fetype(b[0].split('-g')[0], b[2]) for b in a.lhs])

                bc_f2v = np.array(
                    [fnodes(b[0].split('-g')[0], b[1], b[2]) for b in a.lhs]) 

                bc_f2c  = np.array([[b[1],  -1] for b in a.lhs])
                # als need to pad -1 to make sure dimension consistency
                # as it is for now

                bc_fpos = np.array([[b[2],b[2]] for b in a.lhs])
                if 'MPI' in type(a).__name__: # overset bc face
                    bc_instance_idx = np.array([ -1]*bc_fpos.shape[0])
                else:
                    # Instance id of a face with global idx
                    bc_instance_idx = np.array([idx]*bc_fpos.shape[0])

                bfacetypes.append(bc_facetypes)
                bf2v.append(bc_f2v)
                bf2c.append(bc_f2c)
                bfposition.append(bc_fpos)
                # record the face numbers
                bcoffset.append(bc_facetypes.shape[0])
                binstanceid.append(bc_instance_idx)
        
            bfacetypes = np.concatenate(bfacetypes, axis = 0)
            bf2v = np.concatenate(bf2v, axis = 0)
            bf2c = np.concatenate(bf2c, axis = 0)
            bfposition = np.concatenate(bfposition, axis = 0)
            binstanceid = np.concatenate(binstanceid, axis = 0)

        for idx, a in enumerate(int_inters):
            # deal with interior inters
            pd_ftypes_l = np.array(
                [fetype(i[0].split('-g')[0],i[2]) for i in a.lhs if i[3] != 0]
            )
            pd_ftypes_r = np.array(
                [fetype(i[0].split('-g')[0],i[2]) for i in a.rhs if i[3] != 0]
            )
            # sanity check
            if np.all(pd_ftypes_l == pd_ftypes_r) == False:
                raise RuntimeError('Face types inconsistency for elements')

            pd_f2v_l = np.array(
                [fnodes(il[0].split('-g')[0], il[1], il[2]) for il in a.lhs if il[3] != 0]
            )
            pd_f2v_r = np.array(
                [fnodes(ir[0].split('-g')[0], ir[1], ir[2]) for ir in a.rhs if ir[3] != 0]
            )
            # sanity check
            if np.any(np.sort(pd_f2v_l) == np.sort(pd_f2v_r)) == True:
                raise RuntimeError('f2v periodic shared same, not possible')
            
            # Deal with periodic faces
            # Periodic face are merged to be interior faces
            # To feed into tioga, we need to restore them
            pd_f2c_l = np.array([ [i[1], -1] for i in a.lhs if i[3] != 0])
            pd_f2c_r = np.array([ [i[1], -1] for i in a.rhs if i[3] != 0])
            
            pd_fpos_l = np.array([ [i[2],i[2]] for i in a.lhs  if i[3] != 0])
            pd_fpos_r = np.array([ [i[2],i[2]] for i in a.rhs  if i[3] != 0])
            
            # here exists periodic bcs
            if pd_ftypes_l.shape[0] != 0:

                self.int_periodic = pd_ftypes_l.shape[0]

                pdfacetypes = np.concatenate([pd_ftypes_l, pd_ftypes_r], axis = 0)
                pdf2v = np.concatenate([pd_f2v_l,pd_f2v_r], axis = 0)
                pdf2c = np.concatenate([pd_f2c_l,pd_f2c_r], axis = 0)
                pdfposition = np.concatenate([pd_fpos_l,pd_fpos_r], axis=0)
                
                # Instance id of face with global face idx
                pdinstanceid = np.array([idx]*pdfposition.shape[0])

                # exists bfacetypes
                if list(bfacetypes) != []: 
                    bfacetypes = np.concatenate((bfacetypes,pdfacetypes),axis=0)
                    bf2v = np.concatenate((bf2v,pdf2v), axis=0)
                    bf2c = np.concatenate((bf2c,pdf2c), axis=0)
                    bfposition = np.concatenate((bfposition,pdfposition), axis=0)
                    bcinstanceid = np.concatenate((bcinstanceid,pdinstanceid), axis=0)
                else:
                    bfacetypes = pdfacetypes
                    bf2v = pdf2v
                    bf2c = pdf2c
                    bfposition = pdfposition
                    bcinstanceid = pdinstanceid

        # mpi interfaces
        mfacetypes = []
        mf2v = [] 
        mf2c = []
        mfposition = []
        minstanceid = []
        if mpi_inters != []:
            
            mpi_faceidx = []
            mpioffset = [bf2v.shape[0]] if bf2v != [] else [0] # starting idx of mpi faces
            for idx, a in enumerate(mpi_inters):
                mpi_facetypes = np.array( 
                    [fetype(m[0].split('-g')[0],m[2]) for m in a.lhs]
                ) 
                mpi_f2v = np.array(
                    [fnodes(m[0].split('-g')[0],m[1],m[2]) for m in a.lhs]
                ) 
                mpi_f2c  = np.array([ [m[1],  -2] for m in a.lhs]) 
                mpi_fpos = np.array([ [m[2],m[2]] for m in a.lhs])
                
                # Instance id of mpi face with global face idx
                mpi_instance_idx = np.array([idx]*mpi_fpos.shape[0])

                mfacetypes.append(mpi_facetypes)
                mf2v.append(mpi_f2v)
                mf2c.append(mpi_f2c)
                mfposition.append(mpi_fpos)
                minstanceid.append(mpi_instance_idx)
                
                # mpi_idx for later use
                rrank = 0
                mpi_idx = np.arange(mpi_f2v.shape[0]) + mpioffset[idx]
                mpioffset.append(mpioffset[idx]+mpi_f2v.shape[0])
                mpi_faceidx.append([a._rhsrank,mpi_idx])


            mfacetypes = np.concatenate(mfacetypes, axis = 0)
            mf2v = np.concatenate(mf2v, axis = 0)
            mf2c = np.concatenate(mf2c, axis = 0)
            mfposition = np.concatenate(mfposition, axis = 0)
            minstanceid = np.concatenate(minstanceid, axis = 0)
        # interior interfaces
        itfacetypes = [] 
        itf2v = []
        itf2c = []
        itfinfo = []
        itfposition = []
        itfinstanceid = []
        # no way int_inters is an empty list
        for idx, a in enumerate(int_inters):
            # then deal with interior inters
            int_ftypes_l = np.array(
                [fetype(i[0].split('-g')[0],i[2]) for i in a.lhs if i[3] == 0]
            )
            int_ftypes_r = np.array(
                [fetype(i[0].split('-g')[0],i[2]) for i in a.rhs if i[3] == 0]
            )
            # sanity check
            if np.all(int_ftypes_l == int_ftypes_r) == False:
                raise RuntimeError('Face types inconsistency for elements')

            int_facetypes = int_ftypes_l

            int_f2v_l = np.array(
                [fnodes(il[0].split('-g')[0], il[1], il[2]) for il in a.lhs if il[3] == 0]
            )
            int_f2v_r = np.array(
                [fnodes(ir[0].split('-g')[0], ir[1], ir[2]) for ir in a.rhs if ir[3] == 0]
            )

            # sanity check
            if np.all(np.sort(int_f2v_l) == np.sort(int_f2v_r)) == False:
                raise RuntimeError('f2v inconsistency for interior interfaces')

            int_f2v = np.array(int_f2v_l)
            int_f2c = np.array([[il[1], ir[1]] for il,ir in zip(a.lhs,a.rhs)])

            int_fposition = np.array([[il[2], ir[2]] for il,ir in zip(a.lhs,a.rhs)])
            
            int_instance_idx = np.array([idx]*int_fposition.shape[0])

            itfacetypes.append(int_facetypes)
            itf2v.append(int_f2v)
            itf2c.append(int_f2c)
            itfposition.append(int_fposition)
            itfinstanceid.append(int_instance_idx)

        itfacetypes = np.concatenate(itfacetypes, axis = 0)
        itf2v = np.concatenate(itf2v, axis = 0)
        itf2c = np.concatenate(itf2c, axis = 0)
        itfposition = np.concatenate(itfposition, axis = 0)
        itfinstanceid = np.concatenate(itfinstanceid, axis = 0)
        
        if   mf2v != [] and bf2v != []:
            f2v = np.concatenate((bf2v,mf2v,itf2v), axis = 0) 
            f2c = np.concatenate((bf2c,mf2c,itf2c), axis = 0) 
            facetypes = np.concatenate((bfacetypes, mfacetypes, itfacetypes), axis = 0)
            fposition = np.concatenate((bfposition, mfposition, itfposition), axis = 0)
            finstanceid = np.concatenate((binstanceid, minstanceid, itfinstanceid), axis = 0)
        elif mf2v == [] and bf2v != []:
            f2v = np.concatenate((bf2v,itf2v), axis = 0) 
            f2c = np.concatenate((bf2c,itf2c), axis = 0) 
            facetypes = np.concatenate((bfacetypes, itfacetypes), axis = 0)
            fposition = np.concatenate((bfposition, itfposition), axis = 0)
            finstanceid = np.concatenate((binstanceid, itfinstanceid), axis = 0)
        elif mf2v != [] and bf2v == []:
            f2v = np.concatenate((mf2v,itf2v), axis = 0) 
            f2c = np.concatenate((mf2c,itf2c), axis = 0) 
            facetypes = np.concatenate((mfacetypes, itfacetypes), axis = 0)
            fposition = np.concatenate((mfposition, itfposition), axis = 0)
            finstanceid = np.concatenate((minstanceid,itfinstanceid), axis = 0)
        else:
            f2v = np.concatenate((itf2v), axis = 0) 
            f2c = np.concatenate((itf2c), axis = 0) 
            facetypes = np.concatenate((itfacetypes), axis = 0)
            fposition = np.concatenate((itfposition), axis = 0)
            finstanceid = np.concatenate((itfinstanceid), axis = 0)

        nfaces = f2v.shape[0]
        # keep track of nfaces
        self.nfaces = nfaces

        # sort the facetypes 
        srtdfaces = sorted(range(facetypes.shape[0]), key = facetypes.__getitem__)

        # generate c2f from f2c
        for idx, (cidxs, fp) in enumerate(zip(f2c[srtdfaces], fposition[srtdfaces])):
            #for idx, (cidxs, fp) in enumerate(zip(f2c, fposition)):
            lc, rc = cidxs
            lcfpos, rcfpos = fp
            c2f[lc][lcfpos] = idx
            if rc != -1:
                c2f[rc][rcfpos] = idx 
        
        # this is causing the problem do sanity check
        if np.any(c2f == -1):
            raise RuntimeError(f'c2f has -1 entry on Grid {self.gid}')
        
        # query wall nodes and overset nodes
        wallnodes   = np.array([], dtype = self.intdtype)
        overnodes   = np.array([], dtype = self.intdtype)
        wallfaceidx = np.array([], dtype = self.intdtype) # each part has only one 
        overfaceidx = np.array([], dtype = self.intdtype) # each part has only one
        for idx, bc in enumerate(bc_inters):
            ist = sum(bcoffset[:idx+1])
            ied = sum(bcoffset[:idx+2])
            if 'MPI' in type(bc).__name__:
                overnodes = np.unique(
                    [ b for a in f2v[ist:ied] for b in a if b != -1]).astype(self.intdtype)
                overfaceidx = np.array(
                    [srtdfaces[i] for i in np.arange(ist,ied)]).astype(self.intdtype)
            elif 'wall' in bc.type:
                wallnodes = np.unique(
                    [ b for a in f2v[ist:ied] for b in a if b != -1]).astype(self.intdtype)
                wallfaceidx = np.array(
                    [srtdfaces[i] for i in np.arange(ist,ied)]).astype(self.intdtype)
        
        nwallnodes = wallnodes.shape[0] 
        novernodes = overnodes.shape[0] 
        nwallfaces = wallfaceidx.shape[0]
        noverfaces = overfaceidx.shape[0]

        self.nwallnodes = nwallnodes
        self.novernodes = novernodes
        self.nwallfaces = nwallfaces
        self.noverfaces = noverfaces

        # For mpi faces get the information from corresponding cores
        # communicate within the grid communicator
        mpinodes   = np.array([], dtype = self.intdtype)
        mpifaceidx = np.array([], dtype = self.intdtype)
        mpifaces_r = np.array([], dtype = self.intdtype)
        mpifaces_r_rank = np.array([], dtype = self.intdtype)

        if mpi_inters != []:
            MPI_TAG = 2314
            comm = MPI.COMM_WORLD
            # simple copy 
            nbproc  = [a[0] for a in mpi_faceidx]
            recvbuf = [np.empty(a[1].shape[0], dtype=a[1].dtype) for a in mpi_faceidx]
            rranks  = []
            mfaces  = []
            mnodes  = []
            mpifidxinfo = []
            for idx in range(len(mpi_inters)):
                destination = mpi_faceidx[idx][0]
                idxinfo = mpi_faceidx[idx][1]
                # get inorder face idx
                idxinfo = np.array([srtdfaces[i] for i in idxinfo])
                mpifidxinfo.append(idxinfo)

                rhsranks = np.array([destination]*idxinfo.shape[0])
                rranks.append(rhsranks)
                #mfaces.append(idxinfo)
                mnodes.append(np.unique([[b] for a in f2v[idxinfo] for b in a if b != -1]))
                
                # do a datatype check won't hurt
                dtype = MPI.LONG if idxinfo.dtype.name == 'int64' else MPI.INT
                comm.Isend([idxinfo, dtype], destination, MPI_TAG)
            
            status = []
            for idx in range(len(mpi_inters)):
                req = comm.Irecv(recvbuf[idx], source = nbproc[idx], tag = MPI_TAG)
                status.append(req)

            for ireq in status:
                ireq.wait()

            mpifaceidx = np.concatenate(mpifidxinfo, axis = 0).astype(self.intdtype)
            mpinodes = np.concatenate(mnodes, axis = 0).astype(self.intdtype)
            mpifaces_r = np.concatenate(recvbuf, axis = 0).astype(self.intdtype)
            mpifaces_r_rank = np.concatenate(rranks, axis= 0).astype(self.intdtype)
            
            cc=np.zeros(mpifaces_r_rank.shape[0], dtype=self.intdtype)
            for i in range(mpifaces_r_rank.shape[0]):
                d=mpifaces_r_rank[i]                
                cc[i]=self.grrank[d]
            mpifaces_r_rank=cc

        nmpifaces = mpifaceidx.shape[0]
        nmpinodes = mpinodes.shape[0]

        self.nmpifaces = nmpifaces
        self.nmpinodes = nmpinodes

        # flattern c2v c2f f2v by cell face types # need to remove -1
        c2v_flat = []
        c2f_flat = []
        neletypes = [0] + self.nele_etypes.tolist()
        
        vnums = []
        fnums = []
        for i in range(self.nele_etypes.shape[0]):
            ist = sum(neletypes[:i+1])
            ied = sum(neletypes[:i+2])
            
            shapeorder = self.system.ele_map['hex-g{}'.format(self.gid)].basis.nsptsord
            ijk_to_gmsh = structured_hex_to_gmsh(shapeorder)
            c2v_gmsh = np.zeros(c2v[ist:ied].shape)
            c2v_gmsh[:,ijk_to_gmsh] = c2v[ist:ied,0:len(ijk_to_gmsh)]

            flatv = np.array([b for a in c2v_gmsh[ist:ied] for b in a if b != -1])[None,...]
            flatf = np.array([b for a in c2f[ist:ied] for b in a if b != -1])[None,...]
            c2v_flat.append(flatv)
            c2f_flat.append(flatf)

            vnums.append(flatv.shape[1])
            fnums.append(flatf.shape[1])

        maxvnums, maxfnums = max(vnums), max(fnums)
        # get maximum dimension and pad zeros to make sure dimension consistency
        for idx, (flatv, flatf) in enumerate(zip(c2v_flat,c2f_flat)):
            padv = np.array([-1]*(maxvnums-vnums[idx]))[None,...] 
            padf = np.array([-1]*(maxfnums-fnums[idx]))[None,...]
            c2v_flat[idx] = np.concatenate((flatv, padv), axis = 1).astype(self.intdtype)
            c2f_flat[idx] = np.concatenate((flatf, padf), axis = 1).astype(self.intdtype)

        c2v_flat = np.concatenate(c2v_flat, axis = 0).astype(self.intdtype)
        c2f_flat = np.concatenate(c2f_flat, axis = 0).astype(self.intdtype)
        
        # reorder the face, group faces of same type together
        f2v = f2v[srtdfaces]
        # reverse the map and record it 

        unikftypes = np.unique(facetypes)
        unikftypes = np.sort(unikftypes)

        nfacetypes = unikftypes.shape[0]

        self.nfacetypes = nfacetypes

        nftypes = [0] + [sum(facetypes == i) for i in unikftypes]
        
        f2v_flat = []
        vnums = []
        for i in range(nfacetypes):
            ist = sum(nftypes[:i+1])
            ied = sum(nftypes[:i+2])
            flatv = np.array([b for a in f2v[ist:ied] for b in a if b != -1])[None,...]
            f2v_flat.append(flatv)

            vnums.append(flatv.shape[1])

        maxvnums = max(vnums)
        for idx, flatv in enumerate(f2v_flat):
            padv = np.array([-1]*(maxvnums-vnums[idx]))[None,...]
            f2v_flat[idx] = np.concatenate((flatv,padv), axis = 1).astype(self.intdtype)

        f2v_flat = np.concatenate(f2v_flat, axis = 1).astype(self.intdtype)

        nface_ftypes = np.array(nftypes[1:]).astype(self.intdtype)
        self.nface_ftypes = nface_ftypes
        # flatten f2c
        f2c_flat = f2c[srtdfaces].reshape(-1).astype(self.intdtype)

        facetypes = facetypes[srtdfaces]
        faceposition = fposition[srtdfaces]
        f2corg = f2c[srtdfaces]

        finstanceidx = finstanceid[srtdfaces]
        
        # recover overset interfaces are pushed into last 
        for bcint in bc_inters:
            if hasattr(bcint,'_rhsrank'):
                mpi_inters.append(bcint)
                bc_inters.remove(bcint)

        return (c2v_flat, c2f_flat, f2v_flat, f2c_flat, facetypes, faceposition, f2corg,
               overnodes, wallnodes, overfaceidx, wallfaceidx, mpifaceidx, mpifaces_r,
               mpifaces_r_rank, finstanceidx)
        

    def _prepare_griddata(self, tol = 1e-8):
        '''
        Prepate grid data for tioga
        '''
        unikcoords, nnodes_offset, nodesmap = self._form_unique_nodes()
        
        # c2f is only initialized here
        c2v, c2f, celltypes, celloffset = self._form_cell_info(nnodes_offset, nodesmap)
        # I know this is nasty
        (c2v_flat, c2f_flat, f2v_flat, f2c_flat, facetypes, faceposition, f2corg,
        overnodes, wallnodes, overfaceidx, wallfaceidx, mpifaceidx, mpifaces_r,
        mpifaces_r_rank, finstanceidx) = self._form_face_info(c2v, c2f)

        griddata = defaultdict()
        
        griddata['bodytag'] = self.gid # grid id
        griddata['coords'] = unikcoords # unique coordinates 
        
        griddata['nnodes'] = self.nnodes # number of nodes
        griddata['ncells'] = self.ncells # number of cells
        griddata['nfaces'] = self.nfaces # number of faces

        griddata['ncelltypes'] = self.nspt_etypes.shape[0]
        griddata['celltypes'] = celltypes
        griddata['celloffset'] = celloffset
        griddata['nv' ] = self.nspt_etypes # number of nodes for each type
        griddata['nc' ] = self.nele_etypes # number of cells for each type
        griddata['ncf'] = self.nfcs_etypes # number of faces for each type

        griddata['nfacetypes'] = self.nfacetypes # number of face types
        griddata['facetypes'] = facetypes
        griddata['faceposition'] = faceposition
        griddata['f2corg'] = f2corg
        griddata['nf' ] =  self.nface_ftypes 
        griddata['nfv'] =  np.array([9]*self.nfacetypes).astype(self.intdtype) 

        griddata['nwallnodes'] = self.nwallnodes 
        griddata['novernodes'] = self.novernodes 

        griddata['iblank_node'] = np.zeros(unikcoords.shape[0], dtype=self.intdtype)
        griddata['iblank_face'] = np.zeros(self.nfaces, dtype=self.intdtype) 
        griddata['iblank_cell'] = np.zeros(self.ncells, dtype=self.intdtype)
        
        # set eles.cellblank

        griddata['c2v'] = c2v_flat # vconn in tioga: first dimension is ncelltypes
        griddata['c2f'] = c2f_flat # c2f in tioga: first dimension is ncelltypes
        griddata['f2v'] = f2v_flat # fconn in tioga: first dimension is ncelltypes
        griddata['f2c'] = f2c_flat # f2c in tioga: firts dimension is ncelltypes

        griddata['obcnodes'] = overnodes # overset boundary nodes
        griddata['wallnodes'] = wallnodes # wall boundary nodes

        griddata['noverfaces'] = self.noverfaces # number of overset faces
        griddata['nmpifaces'] = self.nmpifaces # number of mpi faces
        griddata['nwallfaces'] = self.nwallfaces # number of wall faces

        griddata['oversetfaces'] = overfaceidx # face idx of overset faces
        griddata['wallfaces'] = wallfaceidx  # face idx of wall faces
        griddata['mpifaces'] = mpifaceidx # face idx of mpi faces
  
        griddata['mpifaces_r'] =  mpifaces_r # face idx of mpi faces on rrank
        griddata['mpifaces_r_rank'] = mpifaces_r_rank # id of rrank of mpi faces
        
        griddata['face_inters_idx'] = finstanceidx
        # first grid always as background grid
        griddata['cuttype'] = 1 if self.gid != 0 else 0

        return griddata

    def _init_overset(self):
        '''
        Convert numpy arrarys into pointers
        '''
        grid = self.griddata

        btag = grid['bodytag']
        xyz = arrayToDblPtr(grid['coords'])
        nnodes = grid['nnodes']
        ncells = grid['ncells']
        nfaces = grid['nfaces']
        
        ncelltypes = grid['ncelltypes']
        nv = arrayToIntPtr(grid['nv'])
        nc = arrayToIntPtr(grid['nc'])
        ncf = arrayToIntPtr(grid['ncf'])

        nfacetypes = grid['nfacetypes']
        nf = arrayToIntPtr(grid['nf'])
        nfv = arrayToIntPtr(grid['nfv'])

        nwallnodes = grid['nwallnodes']
        novernodes = grid['novernodes']

        overnodes = arrayToIntPtr(grid['obcnodes'])
        wallnodes = arrayToIntPtr(grid['wallnodes'])
        
        # new version using ptrAt(array, rowidx, colidx) to check data
        (nrw, nco), ns = grid['c2v'].shape, grid['c2v'].nbytes/grid['c2v'].size
        address, flag = grid['c2v'].__array_interface__['data']
        if flag != False: raise RuntimeError('2D array not in continuous RAM')
        c2v = arrayToDoubleIntPtr(address, nrw,nco, int(ns))
        
        (nrw, nco), ns = grid['c2f'].shape, grid['c2f'].nbytes/grid['c2f'].size
        address, flag = grid['c2f'].__array_interface__['data']
        if flag != False: raise RuntimeError('2D array not in continuous RAM')
        c2f = arrayToDoubleIntPtr(address, nrw,nco, int(ns))

        (nrw, nco), ns = grid['f2v'].shape, grid['f2v'].nbytes/grid['f2v'].size
        address, flag = grid['f2v'].__array_interface__['data']
        if flag != False: raise RuntimeError('2D array not in continuous RAM')
        f2v = arrayToDoubleIntPtr(address, nrw,nco, int(ns))
        
        # need to change
        f2c = arrayToIntPtr(grid['f2c'])

        iblank      = addrToIntPtr(grid['iblank_node'].__array_interface__['data'][0])
        iblank_face = addrToIntPtr(grid['iblank_face'].__array_interface__['data'][0])
        iblank_cell = addrToIntPtr(grid['iblank_cell'].__array_interface__['data'][0])
        
        noverfaces = grid['noverfaces']
        nwallfaces = grid['nwallfaces']
        nmpifaces  = grid['nmpifaces']

        overfaces = arrayToIntPtr(grid['oversetfaces'])
        wallfaces = arrayToIntPtr(grid['wallfaces'])
        mpifaces = arrayToIntPtr(grid['mpifaces'])

        mpifaces_r_fidx = arrayToIntPtr(grid['mpifaces_r'])
        mpifaces_r_rank = arrayToIntPtr(grid['mpifaces_r_rank'])

        gridType = grid['cuttype']
        
        # Begin setting up Tioga class, see /tioga/include/tiogaInterface.h
        tg.tioga_registergrid_data_(btag, nnodes, xyz, iblank,
            nwallnodes, novernodes, wallnodes, overnodes, ncelltypes, nv,
            ncf, nc, c2v)

        tg.tioga_setcelliblank_(iblank_cell)

        tg.tioga_register_face_data_(gridType, f2c, c2f, iblank_face,
            noverfaces, nwallfaces, nmpifaces, overfaces, wallfaces,
            mpifaces, mpifaces_r_rank, mpifaces_r_fidx, nfacetypes, nfv, nf, f2v);
        
        # simplified callbacks
        callbacks = self.callbacks
        
        tg.tioga_set_callbacks_ptr(callbacks)
        tg.tioga_set_highorder_callback_wrapper(callbacks)
        tg.tioga_set_ab_callback_wrapper(callbacks)
        
        self.motion = True
        if self.motion:
            fpdtype = self.system.backend.fpdtype
            grid['gridVel'] = np.array([0,0,0]).astype(fpdtype) # not actually used
            grid['rigidOffset'] = np.array([0,0,0]).astype(fpdtype) 
            grid['rigidRotMat'] = np.array([1,0,0,0,1,0,0,0,1]).astype(fpdtype) 
            grid['rigidPivot']  = np.array([0,0,0]).astype(fpdtype)

            gridV  = addrToFloatPtr(grid['gridVel'].__array_interface__['data'][0])
            offset = addrToFloatPtr(grid['rigidOffset'].__array_interface__['data'][0])
            Rmat   = addrToFloatPtr(grid['rigidRotMat'].__array_interface__['data'][0])
            pivot  = addrToFloatPtr(grid['rigidPivot'].__array_interface__['data'][0])
            
            tg.tioga_register_moving_grid_data(gridV, offset, Rmat, pivot)

        if self.useGpu:

            tg.tioga_set_ab_callback_gpu_wrapper(callbacks)
            backend = self.system.backend
            
            # unique nodes in the mesh
            coords = np.atleast_2d(grid['coords'])
            self.coords_d = backend.matrix(coords.shape,coords)
            
            # reference locations
            self.coords_d_ref = backend.matrix(coords.shape,coords)

            # note here for element coords in tioga (nnodes, dim, neles)
            crds = []
            for _, eles in self.system.ele_map.items():
                eles_ijk = eles.eles.swapaxes(1,2)
                shapeorder = eles.basis.nsptsord
                ijk_to_gmsh = structured_hex_to_gmsh(shapeorder)
                eles_gmsh = np.zeros(eles_ijk.shape)
                eles_gmsh[ijk_to_gmsh,:,:] = eles_ijk[:,:,:]
                crds.append( eles_gmsh.reshape(-1) )

            ecoords = np.concatenate(crds)
            ecoords = np.atleast_2d(ecoords)
            self.ecoords_d = backend.matrix(ecoords.shape, ecoords)
            
            # reference location
            self.ecoords_d_ref = backend.matrix(ecoords.shape, ecoords)

            # allocate fixed size memory chunk
            self.MAX_FRINGE_FACES = 10000
            self.MAX_FPTS = 25
            self.MAX_FRINGE_FPTS = 250000
            self.MAX_UNBLANK_CELLS = 2000
            self.MAX_UPTS = 64

            self.MAX_ORDER = 6
            
            itemsize = np.dtype(self.system.backend.fpdtype).itemsize
            
            self.fringe_coords_d = tg.tg_allocate_device(
              self.MAX_FRINGE_FACES, self.MAX_FPTS, self.ndims, 1, 0, itemsize
            )

            self.fringe_u_fpts_d = tg.tg_allocate_device(
              self.MAX_FRINGE_FACES, self.MAX_FPTS, self.ndims, self.nvars, 0, itemsize
            )

            self.fringe_du_fpts_d = tg.tg_allocate_device(
              self.MAX_FRINGE_FACES, self.MAX_FPTS, self.ndims, self.nvars, 1, itemsize
            )
            
            # memeory for cell unblanking
            self.unblank_coords_d  = tg.tg_allocate_device(
              self.MAX_UNBLANK_CELLS, self.MAX_UPTS, self.ndims, 1, 0, itemsize
            )

            self.unblank_u_d = tg.tg_allocate_device(
              self.MAX_UNBLANK_CELLS, self.MAX_UPTS, self.ndims, self.nvars, 0, itemsize
            )

            self.unblank_du_d = tg.tg_allocate_device(
              self.MAX_UNBLANK_CELLS, self.MAX_UPTS, self.ndims, self.nvars, 1, itemsize
            )
            
            # memory for xi 1d this is particularly for hex
            self.xi1d = tg.tg_allocate_device( self.MAX_ORDER, 1, 1, 1, 0, itemsize )


            self.unblank_ids_ele = OrderedDict()
            self.unblank_ids_loc = OrderedDict()
            for etype in self.system.ele_types:
                self.unblank_ids_ele[etype] = tg.tg_allocate_device_int(
                    self.MAX_UNBLANK_CELLS
                )

                self.unblank_ids_loc[etype] = tg.tg_allocate_device_int(
                    self.MAX_UNBLANK_CELLS
                )

            self.fringe_fpts_d = tg.tg_allocate_device_int(self.MAX_FRINGE_FPTS)
            
            # hint: using tg_print_data to check data for debugging purpose
            grid['xi1d'] = self.xi1d
            grid['coords_gpu'] = self.coords_d
            grid['ele_coords_gpu'] = self.ecoords_d

            grid['fringe_coords_d'] = self.fringe_coords_d
            grid['fringe_u_fpts_d'] = self.fringe_u_fpts_d
            grid['fringe_du_fpts_d'] = self.fringe_du_fpts_d

            grid['unblank_coords_d'] = self.unblank_coords_d
            grid['unblank_u_d'] = self.unblank_u_d
            grid['unblank_du_d'] = self.unblank_du_d
            grid['unblank_ids_ele'] = self.unblank_ids_ele
            grid['unblank_ids_loc'] = self.unblank_ids_loc

            # additional memeory for local coordinates manipulation
            self.Rmat_d = tg.tg_allocate_device(
                9, 1, 1, 1, 0, itemsize
            )
            self.offset_d = tg.tg_allocate_device(
                3, 1, 1, 1, 0, itemsize
            )

            self.pivot_d = tg.tg_allocate_device(
                3, 1, 1, 1, 0, itemsize
            )
            
            # here iblank_cell_d and iblank_face_d are exactly the same as previous??
            tg.tioga_set_device_geo_data(
                addrToFloatPtr(self.coords_d.data),
                addrToFloatPtr(self.ecoords_d.data), 
                addrToIntPtr(int(grid['iblank_cell'].__array_interface__['data'][0])),
                addrToIntPtr(int(grid['iblank_face'].__array_interface__['data'][0]))
            )

            tg.tioga_set_stream_handle(self.cudastream,self.cudaevent)

    # Initial grid preprocessing
    def _preprocess(self):
        tg.tioga_preprocess_grids_()

    # Perform the full domain connectivity
    def _performConnectivity(self):
        tg.tioga_performconnectivity_()

    # Perform just the donor/fringe point connecitivity (using current blanking)
    def performPointConnectivity(self):
        tg.tioga_do_point_connectivity()

    # For high-order codes: First part of unblank procedure (t^{n+1} blanking)
    def unblankPart1(self, motion):
        self.update_transform(motion['Rmat'], motion['pivot'], motion['offset'])
        self.update_adt_transform(motion)
        self.move_flat(motion)
        self.move_nested(motion)
        self.move_on_cpu()
        tg.tioga_unblank_part_1()

    # For high-order codes: Second part of unblank procedure (t^n blanking + union)
    def unblankPart2(self, motion):
        self.update_transform(motion['Rmat'], motion['pivot'], motion['offset'])
        self.update_adt_transform(motion)
        self.move_flat(motion)
        self.move_nested(motion)
        self.move_on_cpu()
        tg.tioga_unblank_part_2(self.nfields)
    
    def update_adt_transform(self, motion):
        fpdtype = self.system.backend.fpdtype
        Rmat = motion['Rmat']
        offset = motion['offset']
        pivot = motion['pivot']
        Rmat = np.array(Rmat).astype(fpdtype)
        offset = np.array(offset).astype(fpdtype)
        pivot = np.array(pivot).astype(fpdtype)
        self.griddata['rigidRotMat'][:] = Rmat[:]
        self.griddata['rigidOffset'][:] = offset[:]
        self.griddata['rigidPivot'][:] = pivot[:]
        # see search.C
        tg.tioga_set_transform(
            addrToFloatPtr(self.griddata['rigidRotMat'].__array_interface__['data'][0]),
            addrToFloatPtr(self.griddata['rigidPivot'].__array_interface__['data'][0]),
            addrToFloatPtr(self.griddata['rigidOffset'].__array_interface__['data'][0]),
            self.ndims
        )
        
    def update_transform(self, Rmat, pivot, offset):
        
        fpdtype = self.system.backend.fpdtype
        Rmat = np.atleast_2d(np.array(Rmat).astype(fpdtype)).reshape(-1,3).reshape(-1)
        pivot  = np.atleast_2d(np.array(pivot ).astype(fpdtype))
        offset = np.atleast_2d(np.array(offset).astype(fpdtype))

        tg.tg_copy_to_device(
            self.Rmat_d, addrToFloatPtr(Rmat.__array_interface__['data'][0]), 
            Rmat.nbytes
        )

        tg.tg_copy_to_device(
            self.offset_d, addrToFloatPtr(offset.__array_interface__['data'][0]),
            offset.nbytes
        )

        tg.tg_copy_to_device(
            self.pivot_d, addrToFloatPtr(pivot.__array_interface__['data'][0]),
            pivot.nbytes
        )

    def move_on_cpu(self):
        tg.tg_copy_to_host(
            int(self.coords_d.data),
            addrToFloatPtr(self.griddata['coords'].__array_interface__['data'][0]),
            self.griddata['coords'].nbytes
        )

    # move grid
    def move_flat(self, motion):
        # calculate the rotation as well as offset
        # face points are update through PyFR kernels
        npts = self.griddata['coords'].shape[0] // self.ndims
        tg.move_grid_flat_wrapper(
            addrToFloatPtr(self.coords_d.data),
            addrToFloatPtr(self.coords_d_ref.data),
            npts, self.ndims, 1.0, 
            addrToFloatPtr(self.Rmat_d),
            addrToFloatPtr(self.offset_d),
            addrToFloatPtr(self.pivot_d),
            3
        )
        # move nested coords on device by types

    def move_nested(self, motion):
        itemsize = self.ecoords_d.itemsize

        offset = 0
        for _, eles in self.system.ele_map.items():
            npts, neles, ndims = eles.eles.shape # second is the number of elements
            tot_item = npts*neles*ndims

            tg.move_grid_nested_wrapper(
                addrToFloatPtr(self.ecoords_d.data+offset),
                addrToFloatPtr(self.ecoords_d_ref.data+offset),
                neles, npts, ndims, 1.0,
                addrToFloatPtr(self.Rmat_d),
                addrToFloatPtr(self.offset_d),
                addrToFloatPtr(self.pivot_d),
                3
            )

            offset = offset+tot_item*itemsize

    def sync_device(self):
        tg.sync_device()

    # Interpolate solution and send/receive all data
    def exchangeSolution(self):
        tg.tioga_dataupdate_ab(self.nfields, 0)

    # Interpolate solution gradient and send/receive all data
    def exchangeGradient(self):
        tg.tioga_dataupdate_ab(self.nfields, 1)

    def exchangeSolutionAMR(self):
        tg.tioga_dataupdate_amr(self.q,self.nfields,1)

    def performAMRConnectivity(self):
        tg.tioga_performconnectivity_amr_()

    def getCallbacks(self):
        return tg.tioga_get_callbacks()
    
    def finish(self):
        tg.tioga_delete_()
    
    def process_restart(self):
        '''
        for restart need to move the grid to restart time instance
        and redo the connectivity stuff
        '''
        runall = self.system.backend.runall
        q1, _ = self.system._queues
        kernels = self.system._kernels
        
        t = self.system.tcurr
        #move the grid 
        motioninfo = self.system.motioninfo
        calc_motion = self.system._calc_motion
        motion = calc_motion(t,t,motioninfo, self.system.backend.fpdtype)

        of = motion['offset']
        R = motion['Rmat']
        pivot = motion['pivot']
        q1 << kernels['eles','updateplocface'](
               t=t, 
               r00 = R[0],  r01 = R[1],  r02 = R[2],
               r10 = R[3],  r11 = R[4],  r12 = R[5],
               r20 = R[6],  r21 = R[7],  r22 = R[8],
               ofx = of[0], ofy = of[1], ofz = of[2],
               pvx = pivot[0], pvy = pivot[1], pvz = pivot[2]
        )

        q1 << kernels['eles','updateploc'](
               t=t, 
               r00 = R[0],  r01 = R[1],  r02 = R[2],
               r10 = R[3],  r11 = R[4],  r12 = R[5],
               r20 = R[6],  r21 = R[7],  r22 = R[8],
               ofx = of[0], ofy = of[1], ofz = of[2],
               pvx = pivot[0], pvy = pivot[1], pvz = pivot[2]
        )
                
        runall([q1])
        
        self.update_transform(motion['Rmat'], motion['pivot'], motion['offset'])
        self.update_adt_transform(motion)
        self.move_flat(motion)
        self.move_nested(motion)
        self.move_on_cpu()
        self.sync_device()

