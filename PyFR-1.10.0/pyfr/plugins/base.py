# -*- coding: utf-8 -*-

import os
import shlex
import numpy as np
from collections import defaultdict

from pytools import prefork

from pyfr.mpiutil import get_comm_rank_root


def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header'):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    # Append the '.csv' extension
    if not fname.endswith('.csv'):
        fname += '.csv'

    # Open for appending
    outf = open(fname, 'a')

    # Output a header if required
    if os.path.getsize(fname) == 0 and cfg.getbool(cfgsect, headerkey, True):
        print(header, file=outf)

    # Return the file
    return outf


class BasePlugin(object):
    name = None
    systems = None
    formulations = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

        # Tolerance for time comparisons
        self.tol = 5*intg.dtmin

        # Initalise our post-action (if any)
        self.postact = None
        self.postactaid = None
        self.postactmode = None

        if self.cfg.hasopt(cfgsect, 'post-action'):
            self.postact = self.cfg.getpath(cfgsect, 'post-action')
            self.postactmode = self.cfg.get(cfgsect, 'post-action-mode',
                                            'blocking')

            if self.postactmode not in {'blocking', 'non-blocking'}:
                raise ValueError('Invalid post action mode')

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError('System {0} not supported by plugin {1}'
                               .format(intg.system.name, self.name))

        # Check that we support this particular integrator formulation
        if intg.formulation not in self.formulations:
            raise RuntimeError('Formulation {0} not supported by plugin {1}'
                               .format(intg.formulation, self.name))

    def __del__(self):
        if self.postactaid is not None:
            prefork.wait(self.postactaid)

    def _invoke_postaction(self, **kwargs):
        comm, rank, root = get_comm_rank_root()

        # If we have a post-action and are the root rank then fire it
        if rank == root and self.postact:
            # If a post-action is currently running then wait for it
            if self.postactaid is not None:
                prefork.wait(self.postactaid)

            # Prepare the command line
            cmdline = shlex.split(self.postact.format(**kwargs))

            # Invoke
            if self.postactmode == 'blocking':
                prefork.call(cmdline)
            else:
                self.postactaid = prefork.call_async(cmdline)

    def __call__(self, intg):
        pass

class RegionMixing(object):

    def __init__(self, intg, cfgsect, *args, **kwargs):
        super().__init__(intg, *args, **kwargs)
        # Output region
        self.mdata=[]
        region = self.cfg.get(cfgsect, 'region', '*')
        # All elements
        if region == '*':
            mdata = self._prepare_mdata_all(intg)
            self.prepare_data = self._prepare_data_all
        # All elements inside a box
        elif ',' in region:
            box = self.cfg.getliteral(cfgsect, 'region')
            mdata = self._prepare_mdata_box(intg, *box)
            self.prepare_data = self._prepare_data_subset
        # All elements on a boundary
        else:
            mdata = self._prepare_mdata_bcs(intg, region)
            self.prepare_data = self._prepare_data_subset

        self.mdata=mdata

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
        
        


    def _prepare_mdata_bcs(self, intg, bcname):
        comm, rank, root = get_comm_rank_root()
        elemap = intg.system.ele_map
        # Get the mesh and prepare the element set dict
        mesh = intg.system.mesh
        eset = defaultdict(list)
        gid, offset = intg.system.gid, intg.system.goffset
        bc = f'bcon_{bcname}_p{intg.rallocs.prank-offset}'

        zipmesh = [mesh] if isinstance(mesh,list) == False else mesh
        tmesh = zipmesh[gid]
        bcranks = comm.gather(bc in tmesh, root=root)
        # Boundary of interest
        
        # Boundary to integrate over
        #bc = 'bcon_{0}_p{1}'.format(suffix, intg.rallocs.prank-offset)
        # See which ranks have the boundary           
        # Ensure the boundary exists
       
        if rank == root and not any(bcranks):
            raise ValueError(f'Boundary {bcname} does not exist')
        mdata,ddata,mavedata= [],[],[]
        if bc in tmesh:            
            for etype, eidx, fidx, flags in tmesh[bc].astype('U4,i4,i1,i2'):
            # Determine which of our elements are on the boundary
            #for etype, eidx in tmesh[bc][['f0', 'f1']].astype('U4,i4'):
                # add gid
                etypem = '{}-g{}'.format(etype,gid)
                eset[etypem].append(eidx)
            

            elemap = intg.system.ele_map            
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


    def _prepare_data_subset(self, intg):
        data = []
        self._darr=[]
        for doff, darr in self._ddata:
            data.append(intg.soln[doff][..., darr])
            data.append(darr)

        return data

    def _prepare_region_data_eset(self, intg, eset):

        comm, rank, root = get_comm_rank_root()
        elemap = intg.system.ele_map
        # Get the mesh and prepare the element set dict
        mesh = intg.system.mesh
        eset = defaultdict(list)
        gid, offset = intg.system.gid, intg.system.goffset
        bc = f'bcon_{bcname}_p{intg.rallocs.prank-offset}'
        zipmesh = [mesh] if isinstance(mesh,list) == False else mesh
        tmesh = zipmesh[gid]
        bcranks = comm.gather(bc in tmesh, root=root)
        # Boundary of interest
        
        # Boundary to integrate over
        #bc = 'bcon_{0}_p{1}'.format(suffix, intg.rallocs.prank-offset)
        # See which ranks have the boundary
         
        # Ensure the boundary exists
 
        if rank == root and not any(bcranks):
            raise ValueError(f'Boundary {bcname} does not exist')
        mdata,ddata= [],[]
        if bc in tmesh:
            for etype, eidx, fidx, flags in tmesh[bc].astype('U4,i4,i1,i2'):
                # Determine which of our elements are on the boundary
                #for etype, eidx in tmesh[bc][['f0', 'f1']].astype('U4,i4'):
                # add gid
                etypem = '{}-g{}'.format(etype,gid)
                eset[etypem].append(eidx)

            self._ele_regions, self._ele_region_data = [], {}
            for etype, eidxs in sorted(eset.items()):
                doff = intg.system.ele_types.index(etype)
                darr = np.unique(eidxs).astype(np.int32)
                self._ele_regions.append((doff, etype, darr))
                self._ele_region_data[f'{etype}_idxs'] = darr

        return mdata
