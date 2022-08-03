# -*- coding: utf-8 -*-

from pyfr.backends.base import ComputeMetaKernel
from pyfr.solvers.base import BaseElements


class BaseAdvectionElements(BaseElements):
    @property
    def _scratch_bufs(self):
        if 'flux' in self.antialias:
            bufs = {'scal_fpts', 'scal_qpts', 'vect_qpts'}
        elif 'div-flux' in self.antialias:
            bufs = {'scal_fpts', 'vect_upts', 'scal_qpts'}
        else:
            bufs = {'scal_fpts', 'vect_upts'}

        if self._soln_in_src_exprs:
            if 'div-flux' in self.antialias:
                bufs |= {'scal_qpts_cpy'}
            else:
                bufs |= {'scal_upts_cpy'}

        return bufs

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        slicem = self._slice_mat
        kernels = self.kernels

        # Register pointwise kernels with the backend
        self._be.pointwise.register(
            'pyfr.solvers.baseadvec.kernels.negdivconf'
        )

        # What anti-aliasing options we're running with
        fluxaa = 'flux' in self.antialias
        divfluxaa = 'div-flux' in self.antialias

        # What the source term expressions (if any) are a function of
        plocsrc = self._ploc_in_src_exprs
        solnsrc = self._soln_in_src_exprs

        # Source term kernel arguments
        srctplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'srcex': self._src_exprs,
            'gid': self.gid
        }

        if self.mvgrid is True: srctplargs['mvars'] = self.mvars

        self.tcurr  = kwargs['tcur']
        self.tstage = kwargs['tstage']

        # Interpolation from elemental points
        for s, neles in self._ext_int_sides:
            if fluxaa or (divfluxaa and solnsrc):
                kernels['disu_' + s] = lambda s=s: self._be.kernel(
                    'mul', self.opmat('M8'), slicem(self.scal_upts_inb, s),
                    out=slicem(self._scal_fqpts, s)
                )
            else:
                kernels['disu_' + s] = lambda s=s: self._be.kernel(
                    'mul', self.opmat('M0'), slicem(self.scal_upts_inb, s),
                    out=slicem(self._scal_fpts, s)
                )

        # Interpolations and projections to/from quadrature points
        if divfluxaa:
            kernels['tdivf_qpts'] = lambda: self._be.kernel(
                'mul', self.opmat('M7'), self.scal_upts_outb,
                out=self._scal_qpts
            )
            kernels['divf_upts'] = lambda: self._be.kernel(
                'mul', self.opmat('M9'), self._scal_qpts,
                out=self.scal_upts_outb
            )

        # First flux correction kernel
        if fluxaa:
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('(M1 - M3*M2)*M10'), self._vect_qpts,
                out=self.scal_upts_outb
            )
        else:
            # f_loc and fn_loc
            kernels['tdivtpcorf'] = lambda: self._be.kernel(
                'mul', self.opmat('M1 - M3*M2'), self._vect_upts,
                out=self.scal_upts_outb
            )

        # Second flux correction kernel where fn_comm functions
        kernels['tdivtconf'] = lambda: self._be.kernel(
            'mul', self.opmat('M3'), self._scal_fpts, out=self.scal_upts_outb,
            beta=1.0
        )

        # Transformed to physical divergence kernel + source term
        if divfluxaa:
            # consider if moving grid if so always need plocupts 
            # to allow for updating coordinates on backend
            if self.mvgrid is True:
                plocqpts = self.ploc_at_ncon('qpts')
                rcpdjacm = self.rcpdjac_at_ncon('qpts')
            else:
                plocqpts = self.ploc_at('qpts') if plocsrc else None
                rcpdjacm = self.rcpdjac_at('qpts')
                
            solnqpts = self._scal_qpts_cpy if solnsrc else None

            if solnsrc:
                kernels['copy_soln'] = lambda: self._be.kernel(
                    'copy', self._scal_qpts_cpy, self._scal_qpts
                )

            kernels['negdivconf'] = lambda: self._be.kernel(
                'negdivconf', tplargs=srctplargs,
                dims=[self.nqpts, self.neles], tdivtconf=self._scal_qpts,
                rcpdjac=rcpdjacm, ploc=plocqpts, u=solnqpts, gclcomp=None
            )
            # update moving grid related kernels
            if self.mvgrid is True:
                self.mvgrid_cls.update_ploc_backend(
                    self,kernels,srctplargs,[self.nqpts,self.neles],
                    plocqpts
                )
                self.mvgrid_cls.update_jaco_backend(
                    self,kernels,srctplargs,
                    [self.nqpts,self.neles],plocqpts,
                    self.smat_at_ncon('qpts'), self.smat_at('qpts'),
                    rcpdjacm, self.rcpdjac_at('qpts')
                )
                self.mvgrid_cls.update_mvel_backend(
                    self,kernels,srctplargs,[self.nqpts,self.neles],
                    plocqpts,self._vect_qpts_mvel
                )
        else:
            # consider if moving grid if so always need plocupts 
            # to allow for updating coordinates on backend
            if self.mvgrid is True:
                plocupts = self.ploc_at_ncon('upts') 
                rcpdjacm = self.rcpdjac_at_ncon('upts')
                plocfpts = self.ploc_at_ncon('fpts')
            else:
                plocupts = self.ploc_at('upts') if plocsrc else None
                rcpdjacm = self.rcpdjac_at('upts')

            solnupts = self._scal_upts_cpy if solnsrc else None

            if solnsrc:
                kernels['copy_soln'] = lambda: self._be.kernel(
                    'copy', self._scal_upts_cpy, self.scal_upts_inb
                )

            kernels['negdivconf'] = lambda: self._be.kernel(
                'negdivconf', tplargs=srctplargs,
                dims=[self.nupts, self.neles], tdivtconf=self.scal_upts_outb,
                rcpdjac=rcpdjacm, ploc=plocupts, u=solnupts, 
                s=self.scal_upts_inb, gclcomp=self._scal_upts_gcl
            )
            # update moving grid related kernels
            if self.mvgrid is True:
                self.mvgrid_cls.update_ploc_backend(
                    self,kernels,srctplargs,[self.nupts,self.neles],
                    plocupts
                )
                # directly evaluate the mvel on fpts
                # for overset also need to update the fpts
                self.mvgrid_cls.update_ploc_face_backend(
                    self,kernels,srctplargs,[self.nfpts,self.neles],
                    plocfpts
                )

                self.mvgrid_cls.update_jaco_backend(
                    self,kernels,srctplargs,
                    [self.nupts,self.neles], plocupts,
                    self.smat_at_ncon('upts'), self.smat_at('upts'),
                    rcpdjacm,self.rcpdjac_at('upts')
                )

                self.mvgrid_cls.rotate_jaco_backend(
                    self,kernels,srctplargs,
                    [self.nupts,self.neles], plocupts,
                    self.smat_at_ncon('upts'), self.smat_at('upts'),
                    rcpdjacm,self.rcpdjac_at('upts')
                )

                self.mvgrid_cls.update_mvel_backend(
                    self,kernels,srctplargs,[self.nupts,self.neles],
                    plocupts,self._vect_upts_mvel
                )

                self.mvgrid_cls.update_mvel_face_backend(
                    self,kernels,srctplargs,[self.nfpts,self.neles],
                    plocfpts,self._vect_fpts_mvel
                )

                self.mvgrid_cls.interpolate_mvel_smat_kernel(
                    self,kernels,slicem
                )
                # here need a kernel to add u*(∂(|J|ξ_t)/∂ξ+∂(|J|η_t)/∂η+∂(|J|ζ_t)/∂ζ)
                # reuse bufs to avoid ram  allocation
                # note this part is more like adding source terms
                self.mvgrid_cls.update_gcl_component(
                    self,kernels,srctplargs,[self.nupts,self.neles],
                    self.smat_at_ncon('upts'),self._vect_upts_mvel
                )

        # In-place solution filter
        if self.cfg.getint('soln-filter', 'nsteps', '0'):
            def filter_soln():
                mul = self._be.kernel(
                    'mul', self.opmat('M11'), self.scal_upts_inb,
                    out=self._scal_upts_temp
                )
                copy = self._be.kernel(
                    'copy', self.scal_upts_inb, self._scal_upts_temp
                )

                return ComputeMetaKernel([mul, copy])

            kernels['filter_soln'] = filter_soln
