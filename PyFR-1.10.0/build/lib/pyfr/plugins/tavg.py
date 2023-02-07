# -*- coding: utf-8 -*-

import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin, RegionMixing
from pyfr.nputil import npeval
from pyfr.writers.native import NativeWriter
from pyfr.mpiutil import get_comm_rank_root

from collections import defaultdict


class TavgPlugin(BasePlugin,RegionMixing):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        # Underlying elements class
        self.elementscls = intg.system.elementscls
        comm, rank, root = get_comm_rank_root()
        # Expressions to time average
        c = self.cfg.items_as('constants', float)
        self.exprs = [(k, self.cfg.getexpr(cfgsect, k, subs=c))
                      for k in self.cfg.items(cfgsect)
                      if k.startswith('avg-')]
        self._prepare_exprs()
        # Gradient pre-processing
        self._init_gradients(intg)
        self._ele_regions=[] 
        self._ele_region_data=np.array([])
        # Save a reference to the physical solution point locations
        self.plocs = intg.system.ele_ploc_upts

        # Element info and backend data type
        einfo = zip(intg.system.ele_types, intg.system.ele_shapes)
        fpdtype = intg.backend.fpdtype
        region = self.cfg.get(cfgsect, 'region', '*')
        if region=='*':
        # Output metadata
            mdata = [(f'tavg_{etype}', (nupts, len(self.exprs), neles), fpdtype)
                    for etype, (nupts, _, neles) in einfo]
        else:
             mdata=self._prepare_region_data_eset(intg,region)
        # Output base directory and name
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        self._writer = NativeWriter(intg, mdata, basedir, basename)

        # Time averaging parameters
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.tout_last = intg.tcurr

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dtout)

        # Time averaging state
        self.prevt = intg.tcurr
        self.prevex = self._eval_exprs(intg)
        self.accmex = [np.zeros_like(p) for p in self.prevex]

    def _prepare_exprs(self):
        cfg, cfgsect = self.cfg, self.cfgsect
        c = self.cfg.items_as('constants', float)
        self.anames, self.aexprs = [], []
        self.outfields, self.fexprs = [], []
        # Iterate over accumulation expressions first
        for k in cfg.items(cfgsect):
            if k.startswith('avg-'):
                self.anames.append(k[4:])
                self.aexprs.append(cfg.getexpr(cfgsect, k, subs=c))
                self.outfields.append(k)
        # Followed by any functional expressions
        for k in cfg.items(cfgsect):
            if k.startswith('fun-avg-'):
                self.fexprs.append(cfg.getexpr(cfgsect, k, subs=c))
                self.outfields.append(k)

    def _init_gradients(self, intg):
        # Determine what gradients, if any, are required
        gradpnames = set()
        for ex in self.aexprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))
        privarmap = self.elementscls.privarmap[self.ndims]
        self._gradpinfo = [(pname, privarmap.index(pname))
                        for pname in gradpnames]
    '''
    def _init_gradients(self, intg):
        # Determine what gradients, if any, are required
        self._gradpnames = gradpnames = set()
        for k, ex in self.exprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        # If gradients are required then form the relevant operators
        if gradpnames:
            self._gradop, self._rcpjact = [], []

            for eles in intg.system.ele_map.values():
                self._gradop.append(eles.basis.m4)

                # Get the smats at the solution points
                smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                # Get |J|^-1 at the solution points
                rcpdjac = eles.rcpdjac_at_np('upts')

                # Product to give J^-T at the solution points
                self._rcpjact.append(smat*rcpdjac)
    '''
                
    def _eval_exprs(self, intg):
        exprs = []

        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        for idx, etype, rgn in self._ele_regions:
            soln = intg.soln[idx][..., rgn].swapaxes(0, 1)
            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln, self.cfg)
            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))
            # Prepare any required gradients
            if self._gradpinfo:
                # Compute the gradients
                grad_soln = np.rollaxis(intg.grad_soln[idx], 2)[..., rgn]
                # Transform from conservative to primitive gradients
                pgrads = self.elementscls.grad_con_to_pri(soln, grad_soln,
                                                        self.cfg)
                # Add them to the substitutions dictionary
                for pname, idx in self._gradpinfo:
                    for dim, grad in zip('xyz', pgrads[idx]):
                        subs[f'grad_{pname}_{dim}'] = grad
            # Evaluate the expressions
            exprs.append([npeval(v, subs) for v in self.aexprs])
        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

        '''
        # Iterate over each element type in the simulation
        for i, (soln, ploc) in enumerate(zip(intg.soln, self.plocs)):
            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln.swapaxes(0, 1),
                                                 self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))
            subs.update(zip('xyz', ploc.swapaxes(0, 1)))

            # Compute any required gradients
            if self._gradpnames:
                # Gradient operator and J^-T matrix
                gradop, rcpjact = self._gradop[i], self._rcpjact[i]
                nupts = gradop.shape[1]

                for pname in self._gradpnames:
                    psoln = subs[pname]

                    # Compute the transformed gradient
                    tgradpn = gradop @ psoln
                    tgradpn = tgradpn.reshape(self.ndims, nupts, -1)

                    # Untransform this to get the physical gradient
                    gradpn = np.einsum('ijkl,jkl->ikl', rcpjact, tgradpn)
                    gradpn = gradpn.reshape(self.ndims, nupts, -1)

                    for dim, grad in zip('xyz', gradpn):
                        subs['grad_{0}_{1}'.format(pname, dim)] = grad

            # Evaluate the expressions
            exprs.append([npeval(v, subs) for k, v in self.exprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]
        '''
    def _prepare_region_data_eset(self, intg,bcname):

        comm, rank, root = get_comm_rank_root()
        elemap = intg.system.ele_map
        self.rank=rank
        # Get the mesh and prepare the element set dict
        mesh = intg.system.mesh
        eset = defaultdict(list)
        self._gid, offset = intg.system.gid, intg.system.goffset
        gid=self._gid
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
        mdata,ddata= [],{}
        if bc in tmesh:
            einfo = zip(intg.system.ele_types, intg.system.ele_shapes)

            for etype, eidx, fidx, flags in tmesh[bc].astype('U4,i4,i1,i2'):
            # Determine which of our elements are on the boundary
            #for etype, eidx in tmesh[bc][['f0', 'f1']].astype('U4,i4'):
                # add gid
                etypem = '{}-g{}'.format(etype,gid)
                eset[etypem].append(eidx)
            
            for etype, eidxs in sorted(eset.items()):
                
                neidx = len(eidxs)
                shape = (elemap[etype].nupts, elemap[etype].nvars, neidx)
                mdata.append((f'tavg_{etype}', 
                    (elemap[etype].nupts, len(self.exprs), neidx), intg.backend.fpdtype))
                mdata.append((f'tavg_{etype}_idxs', (neidx,), np.int32))
                doff = intg.system.ele_types.index(etype)
                darr = np.unique(eidxs).astype(np.int32)
                self._ele_regions.append((doff, etype, darr))
                self._ele_region_data=darr
        return mdata


    def __call__(self, intg):
        tdiff = intg.tcurr - self.tout_last
        dowrite = tdiff >= self.dtout - self.tol
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            # Evaluate the time averaging expressions
            currex = self._eval_exprs(intg)
            
            # Accumulate them; always do this even when just writing
            for a, p, c in zip(self.accmex, self.prevex, currex):
                a += 0.5*(intg.tcurr - self.prevt)*(p + c)

            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex

            if dowrite:
                # Normalise
                accmex = [a / tdiff for a in self.accmex]

                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields',
                          ','.join(k for k, v in self.exprs))
                stats.set('tavg', 'tstart', self.tout_last)
                stats.set('tavg', 'tend', intg.tcurr)
                intg.collect_stats(stats)

                metadata = dict(intg.cfgmeta,
                                stats=stats.tostr(),
                                mesh_uuid=intg.mesh_uuid)
                
                data=[]
                data=accmex
                if self._ele_region_data.size:
                    data.append(self._ele_region_data)
                #print('\n accmex',self.rank,data)
                self._writer.write(data, metadata, intg.tcurr)
                mesh = intg.system.mesh
                zipmesh = [mesh] if isinstance(mesh,list) == False else mesh
                # If a post-action has been registered then invoke it
                # support multiple meshes
                self._invoke_postaction(mesh=[ a.fname for a in zipmesh], soln=accmex,
                        t=intg.tcurr)
                self.tout_last = intg.tcurr
                #self.accmex = [np.zeros_like(a) for a in accmex]
                data.clear()