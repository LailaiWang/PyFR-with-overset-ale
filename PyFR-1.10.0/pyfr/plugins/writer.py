# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin
from pyfr.writers.native import NativeWriter


class WriterPlugin(BasePlugin):
    name = 'writer'
    systems = ['*']
    formulations = ['dual', 'std']
    
    # writer will write the solutions on multiple meshes into one file
    # for visualization or post-processing, solutions will be reassigned 
    # into different files

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Output region
        region = self.cfg.get(cfgsect, 'region', '*')

        # All elements
        if region == '*':
            mdata = self._prepare_mdata_all(intg)
            self._prepare_data = self._prepare_data_all
        # All elements inside a box
        elif ',' in region:
            box = self.cfg.getliteral(cfgsect, 'region')
            mdata = self._prepare_mdata_box(intg, *box)
            self._prepare_data = self._prepare_data_subset
        # All elements on a boundary
        else:
            mdata = self._prepare_mdata_bcs(intg, region)
            self._prepare_data = self._prepare_data_subset

        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')
        self._writer = NativeWriter(intg, mdata, basedir, basename)

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then write out the initial solution
        if not intg.isrestart:
            self.tout_last -= self.dt_out
            self(intg)

    def _prepare_mdata_all(self, intg):
        
        if intg.system.overset is True:
            # Element info and backend data type
            einfo1  = zip(intg.system.ele_types, intg.system.ele_shapes)
            einfo2  = zip(intg.system.ele_types, intg.system.ele_shapes)
            fpdtype = intg.backend.fpdtype
            # output metadata
            mdata = [(f'soln_{etype}', shape, fpdtype) for etype, shape in einfo1]
            # add information of cell blanking
            mdata = mdata + [(f'soln_{etype}_blank', shape[2], np.int32) for etype, shape in einfo2]
            return mdata
        else:
            # Element info and backend data type
            einfo  = zip(intg.system.ele_types, intg.system.ele_shapes)
            fpdtype = intg.backend.fpdtype
            # Output metadata
            return [(f'soln_{etype}', shape, fpdtype) for etype, shape in einfo]

    def _prepare_mdata_box(self, intg, x0, x1):
        eset = {}

        for etype in intg.system.ele_types:
            #pts = intg.system.mesh[f'spt_{etype}_p{intg.rallocs.prank}']
            pts = intg.system.mesh[f'spt_{etype}_p{intg.rallocs.prank-intg.system.goffset}']
            pts = np.moveaxis(pts, 2, 0)

            # Determine which points are inside the box
            inside = np.ones(pts.shape[1:], dtype=np.bool)
            for l, p, u in zip(x0, pts, x1):
                inside &= (l <= p) & (p <= u)

            if np.sum(inside):
                eset[etype] = np.any(inside, axis=0).nonzero()[0]

        return self._prepare_eset(intg, eset)

    def _prepare_mdata_bcs(self, intg, bcname):
        comm, rank, root = get_comm_rank_root()

        # Get the mesh and prepare the element set dict
        mesh = intg.system.mesh
        eset = defaultdict(list)
        
        zipmesh = [mesh] if isinstance(mesh,list) == False else mesh

        # Boundary of interest
        bc = f'bcon_{bcname}_p{intg.rallocs.prank}'

        bc_chk = [ bc in amesh.keys() for amesh in zipmesh]
        chk = sum(bool(x) for x in bc_chk)
        if chk>1: 
            raise RuntimeError('Boundary {0}  appears in multiple grid'.format(suffix))
        elif chk == 1:
            gid = bc_chk.index(True)
        else:
            gid = 0
        tmesh = zipmesh[gid]

        # Ensure the boundary exists
        bcranks = comm.gather(bc in tmesh, root=root)
        if rank == root and not any(bcranks):
            raise ValueError(f'Boundary {bcname} does not exist')

        if bc in tmesh:
            # Determine which of our elements are on the boundary
            for etype, eidx in tmesh[bc][['f0', 'f1']].astype('U4,i4'):
                # add gid
                etypem = '{}-g{}'.format(etype,gid)
                eset[etypem].append(eidx)

        return self._prepare_eset(intg, eset)

    def _prepare_eset(self, intg, eset):
        elemap = intg.system.ele_map

        mdata, ddata = [], []
        for etype, eidxs in sorted(eset.items()):
            neidx = len(eidxs)
            shape = (elemap[etype].nupts, elemap[etype].nvars, neidx)

            mdata.append((f'soln_{etype}', shape, intg.backend.fpdtype))
            mdata.append((f'soln_{etype}_idxs', (neidx,), np.int32))

            doff = intg.system.ele_types.index(etype)
            darr = np.unique(eidxs).astype(np.int32)

            ddata.append((doff, darr))

        # Save ddata for later use
        self._ddata = ddata

        return mdata

    def _prepare_data_all(self, intg):
        if intg.system.overset is True:
            iblank_cell = intg.system.oset.griddata['iblank_cell']
            # consider multiple type of elements here
            etypeoff = [0]
            neles = 0
            for eshape in intg.system.ele_shapes:
                neles = neles + eshape[2]
                etypeoff.append(neles)
            
            data = []
            for idx, soln in enumerate(intg.soln):
                data.append(soln)
                data.append(iblank_cell[etypeoff[idx]:etypeoff[idx+1]])
            return data
        else:
            return intg.soln

    def _prepare_data_subset(self, intg):
        data = []

        for doff, darr in self._ddata:
            data.append(intg.soln[doff][..., darr])
            data.append(darr)

        return data

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        # Prepare the metadata
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)

        # Prepare the data itself
        data = self._prepare_data(intg)

        # Write out the file
        solnfname = self._writer.write(data, metadata, intg.tcurr)
        
        mesh = intg.system.mesh
        zipmesh = [mesh] if isinstance(mesh,list) == False else mesh
        # If a post-action has been registered then invoke it
        # support multiple meshes
        self._invoke_postaction(mesh=[ a.fname for a in zipmesh], soln=solnfname,
                                t=intg.tcurr)

        # Update the last output time
        self.tout_last = intg.tcurr
