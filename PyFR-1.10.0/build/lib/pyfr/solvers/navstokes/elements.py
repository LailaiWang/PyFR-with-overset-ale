# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionElements
from pyfr.solvers.euler.elements import BaseFluidElements


class NavierStokesElements(BaseFluidElements, BaseAdvectionDiffusionElements):
    # Use the density field for shock sensing
    shockvar = 'rho'

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.tflux')

        shock_capturing = self.cfg.get('solver', 'shock-capturing')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        if visc_corr not in {'sutherland', 'none'}:
            raise ValueError('Invalid viscosity-correction option')

        tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                       shock_capturing=shock_capturing, visc_corr=visc_corr,
                       mvgrid=self.mvgrid,
                       c=self.cfg.items_as('constants', float))

        if 'flux' in self.antialias:
            smatm = self.smat_at('qpts') if self.mvgrid is False else self.smat_at_ncon('qpts')
            self.kernels['tdisf'] = lambda: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[self.nqpts, self.neles],
                u=self._scal_qpts, smats=smatm,
                f=self._vect_qpts, artvisc=self.artvisc, mvel=self._vect_qpts_mvel
            )
        else:
            smatm = self.smat_at('upts') if self.mvgrid is False else self.smat_at_ncon('upts')
            self.kernels['tdisf'] = lambda: self._be.kernel(
                'tflux', tplargs=tplargs, dims=[self.nupts, self.neles],
                u=self.scal_upts_inb, smats=smatm,
                f=self._vect_upts, artvisc=self.artvisc, mvel =self._vect_upts_mvel
            )
