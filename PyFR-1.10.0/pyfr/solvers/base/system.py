# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict
import itertools as it
import re

from pyfr.inifile import Inifile
from pyfr.shapes import BaseShape
from pyfr.util import proxylist, subclasses
from pyfr.solvers.overset.overset import Overset
from pyfr.solvers.moving_grid.movgrid import motion_exprs, calc_motion

class BaseSystem(object):
    elementscls = None
    intinterscls = None
    mpiinterscls = None
    bbcinterscls = None

    # Number of queues to allocate
    _nqueues = None

    # Nonce sequence
    _nonce_seq = it.count()

    def __init__(self, backend, rallocs, mesh, initsoln, nregs, cfg, **kwargs):
        self.backend = backend
        self.mesh = mesh
        self.cfg = cfg

        # get the information of t
        self.tcurr  = kwargs['tcur']
        self.dtcurr = kwargs['dtcur']
        self.tstage = kwargs['tstage']

        self.gid, self.goffset, self.ngrids = self._query_grid_id(rallocs)

        self.mvgrid, self.overset  = False, False

        if 'overset' in cfg.sections(): self.overset = True
        if 'moving-object' in cfg.sections():
            self.mvgrid = True
            mtype = cfg.get('moving-object','type')
            if mtype != 'rigid':
                raise RuntimeError('PyFR only support rigid motion')
            gridmotion = cfg.get('moving-object','grid-{}'.format(self.gid))
            if self.overset is True:
                if self.gid == 0 and gridmotion != 'static':
                    raise RuntimeError('Grid 0 must be static/background')

        
        # Obtain a nonce to uniquely identify this system
        nonce = str(next(self._nonce_seq))

        # Load the elements
        # if there exists multiple meshes merge them toghter
        eles, elemap = self._load_eles(rallocs, mesh, initsoln, nregs, nonce)
        backend.commit()

        # Retain the element map; this may be deleted by clients
        self.ele_map = elemap

        # Get the banks, types, num DOFs and shapes of the elements
        self.ele_banks = list(eles.scal_upts_inb)
        self.ele_types = list(elemap)
        self.ele_ndofs = [e.neles*e.nupts*e.nvars for e in eles]
        self.ele_shapes = [(e.nupts, e.nvars, e.neles) for e in eles]

        # Get all the solution point locations for the elements
        self.ele_ploc_upts = [e.ploc_at_np('upts') for e in eles]

        # I/O banks for the elements
        self.eles_scal_upts_inb = eles.scal_upts_inb
        self.eles_scal_upts_outb = eles.scal_upts_outb

        # Save the number of dimensions and field variables
        self.ndims = eles[0].ndims
        self.nvars = eles[0].nvars

        # Load the interfaces
        int_inters = self._load_int_inters(rallocs, mesh, elemap)
        mpi_inters = self._load_mpi_inters(rallocs, mesh, elemap)
        bc_inters = self._load_bc_inters(rallocs, mesh, elemap)
        backend.commit()
        
        # for debugging purpose
        self._int_inters = int_inters
        self._mpi_inters = mpi_inters
        
        # Prepare the queues and kernels
        self._gen_queues()
        self._gen_kernels(eles, int_inters, mpi_inters, bc_inters)
        backend.commit()

        # Save the BC interfaces, but delete the memory-intensive elemap
        self._bc_inters = bc_inters
        del bc_inters.elemap
        
        # save the motion info
        if self.mvgrid is True:
            self.motioninfo = motion_exprs(self.cfg, self.gid)
            self._calc_motion = calc_motion
            # motion = self._calc_motion(1.0,1.0,self.motioninfo, self.backend.fpdtype)

        if self.overset is True:
            self.oset = Overset(self,rallocs) 

    def _compute_int_offsets(self, rallocs, mesh, gid, offset):
        lhsprank = rallocs.prank
        intoffs = defaultdict(lambda: 0)

        for rhsprank in rallocs.prankconn[lhsprank]:
            interarr = mesh['con_p{0}p{1}'.format(lhsprank-offset, rhsprank-offset)]
            interarr = interarr[['f0', 'f1']].astype('U4,i4').tolist()

            for etype, eidx in interarr:
                etypem = etype+'-g{}'.format(gid)
                intoffs[etypem] = max(eidx + 1, intoffs[etypem])

        return intoffs
        
    def _query_grid_id(self, rallocs):
        grankrange = rallocs.grankrange
        chk = [rallocs.prank < x for x in grankrange]
        # sanity check 
        if any(chk) == False:
            raise RuntimeError('Current {} rank not in grid rankrange'
                               .format(rallocs.prank))

        gid = chk.index(True)
        
        ngrids = len(self.mesh) if isinstance(self.mesh,list) else 1

        return gid, rallocs.grankoffsets[gid], ngrids

    def _load_eles(self, rallocs, mesh, initsoln, nregs, nonce):
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        zipmesh  = [mesh] if isinstance(mesh,list) == False else mesh

        gid      = self.gid
        amesh    = zipmesh[gid]
        goffset  = self.goffset

        # for multiple grids, grid ID is added to ele type eg. hex-g1
        elemap   = OrderedDict()
        for f in amesh:
            m = re.match('spt_(.+?)_p{0}$'.format(rallocs.prank-goffset), f)
            if m:
                # Original element type
                t  = m.group(1)
                # Element type with grid id
                tm = t+'-g{}'.format(gid)
                elemap[tm] = self.elementscls(basismap[t], amesh[f], self.cfg, gid)
        
        # Construct a proxylist to simplify collective operations
        eles = proxylist(elemap.values())

        # Set the initial conditions
        if initsoln:
            # Load the config and stats files from the solution
            solncfg = Inifile(initsoln['config'])
            solnsts = Inifile(initsoln['stats'])

            # restart
            self.restart = True

            # Get the names of the conserved variables (fields)
            solnfields = solnsts.get('data', 'fields', '')
            currfields = ','.join(eles[0].convarmap[eles[0].ndims])

            # Ensure they match up
            if solnfields and solnfields != currfields:
                raise RuntimeError('Invalid solution for system')

            # Process the solution
            for etype, ele in elemap.items():
                soln = initsoln['soln_{0}_p{1}'.format(etype, rallocs.prank)]
                ele.set_ics_from_soln(soln, solncfg)
        else:
            self.restart = False
            eles.set_ics_from_cfg()

        # Compute the index of first strictly interior element
        intoffs = self._compute_int_offsets(rallocs, amesh, gid, goffset)

        # Allocate these elements on the backend
        for etype, ele in elemap.items():
            # add information of time stepper
            ele.set_backend(self.backend, nregs, nonce, intoffs[etype],
                   tcur = self.tcurr,dtcur = self.dtcurr, tstage=self.tstage)

        return eles, elemap

    def _load_int_inters(self, rallocs, mesh, elemap):
        key = 'con_p{0}'.format(rallocs.prank-self.goffset)

        zipmesh = [mesh] if isinstance(mesh,list) == False else mesh
        
        amesh = zipmesh[self.gid]
        # lhs, rhs, face grid idx
        lhs, rhs = amesh[key].astype('U4,i4,i1,i2').tolist()
        # add the grid id 
        for idx,(ilhs, irhs) in enumerate(zip(lhs, rhs)):
            ilhs = ('{}-g{}'.format(ilhs[0],self.gid),ilhs[1],ilhs[2],ilhs[3])
            irhs = ('{}-g{}'.format(irhs[0],self.gid),irhs[1],irhs[2],irhs[3])
            lhs[idx], rhs[idx] = ilhs, irhs

        int_inters = self.intinterscls(self.backend, lhs, rhs, elemap,
                                       self.cfg)
        
        # Although we only have a single internal interfaces instance
        # we wrap it in a proxylist for consistency
        return proxylist([int_inters])

    def _load_mpi_inters(self, rallocs, mesh, elemap):
        lhsprank  = rallocs.prank
        prankconn = rallocs.prankconn
        zipmesh   = [mesh] if isinstance(mesh,list) == False else mesh
        gid       = self.gid
        amesh     = zipmesh[gid]
        offset    = self.goffset

        mpi_inters = proxylist([])
        # generate a class instance for each mpiface
        for rhsprank in prankconn[lhsprank]:
            rhsmrank = rallocs.pmrankmap[rhsprank]
            interarr = amesh['con_p{0}p{1}'.format(lhsprank-offset, rhsprank-offset)]
            interarr = interarr.astype('U4,i4,i1,i2').tolist()

            # add grid information into interarr
            for idx, finfo in enumerate(interarr):
                iint = finfo
                iint = ('{}-g{}'.format(iint[0],gid),iint[1],iint[2],iint[3])
                interarr[idx] = iint

            mpiiface = self.mpiinterscls(self.backend, interarr, rhsmrank,
                                         rallocs, elemap, self.cfg)
            mpi_inters.append(mpiiface)
        # additionally need to check if there is overset type boundaries
        # there are artificial boundaries
        cnt_overset_bc = 0
        for f in amesh:
            m = re.match('bcon_overset_(.+?)_p{0}$'.format(rallocs.prank-offset), f)
            if m:
                cnt_overset_bc = cnt_overset_bc + 1
                # Get the interface
                interarr = amesh[f].astype('U4,i4,i1,i2').tolist()
                for idx, finfo in enumerate(interarr):
                    iint = finfo
                    iint = ('{}-g{}'.format(iint[0],gid),iint[1],iint[2],iint[3])
                    interarr[idx] = iint
                # for this type of problem set rhsrank to None
                mpiiface = self.mpiinterscls(self.backend, interarr, None,
                                             rallocs, elemap, self.cfg)
                mpi_inters.append(mpiiface)
        
        if cnt_overset_bc >1 : raise RuntimeError('Each partition should only have one overset type bc')

        return mpi_inters

    def _load_bc_inters(self, rallocs, mesh, elemap):
        bccls = self.bbcinterscls
        bcmap = {b.type: b for b in subclasses(bccls, just_leaf=True)}

        zipmesh  = [mesh] if isinstance(mesh,list) == False else mesh
        gid      = self.gid
        amesh    = zipmesh[gid]
        offset   = self.goffset

        bc_inters = proxylist([])
        for f in amesh:
            m = re.match('bcon_(.+?)_p{0}$'.format(rallocs.prank-offset), f)
            # skip the overset interface
            if m and 'overset' not in m.group(1):
                # Get the region name
                rgn = m.group(1)

                # Determine the config file section
                cfgsect = 'soln-bcs-%s' % rgn

                # Get the interface
                interarr = amesh[f].astype('U4,i4,i1,i2').tolist()
                for idx, finfo in enumerate(interarr):
                    iint = finfo
                    iint = ('{}-g{}'.format(iint[0],gid),iint[1],iint[2],iint[3])
                    interarr[idx] = iint

                # Instantiate
                bcclass = bcmap[self.cfg.get(cfgsect, 'type')]
                bciface = bcclass(self.backend, interarr, elemap, cfgsect,
                                  self.cfg)
                bc_inters.append(bciface)

        return bc_inters

    def _gen_queues(self):
        self._queues = [self.backend.queue() for i in range(self._nqueues)]

    def _gen_kernels(self, eles, iint, mpiint, bcint):
        self._kernels = kernels = defaultdict(proxylist)

        provnames = ['eles', 'iint', 'mpiint', 'bcint']
        provobjs = [eles, iint, mpiint, bcint]

        for pn, pobj in zip(provnames, provobjs):
            for kn, kgetter in it.chain(*pobj.kernels.items()):
                if not kn.startswith('_'):
                    kernels[pn, kn].append(kgetter())

    def rhs(self, t, uinbank, foutbank):
        pass

    def filt(self, uinoutbank):
        self.eles_scal_upts_inb.active = uinoutbank

        self._queues[0] % self._kernels['eles', 'filter_soln']()

    def ele_scal_upts(self, idx):
        return [eb[idx].get() for eb in self.ele_banks]
