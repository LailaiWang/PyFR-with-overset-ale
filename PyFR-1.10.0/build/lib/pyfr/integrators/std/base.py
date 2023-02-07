# -*- coding: utf-8 -*-

from pyfr.integrators.base import BaseIntegrator
from pyfr.integrators.base import BaseCommon
from pyfr.util import proxylist


class BaseStdIntegrator(BaseCommon, BaseIntegrator):
    formulation = 'std'

    def __init__(self, backend, systemcls, rallocs, mesh, initsoln, cfg):
        super().__init__(backend, rallocs, mesh, initsoln, cfg)

        # Determine the amount of temp storage required by this method
        self.nregs = self._stepper_nregs

        self.system = systemcls(backend, rallocs, mesh, initsoln,
                                nregs=self.nregs, cfg=cfg, 
                                tcur = self.tcurr, dtcur = self._dt, 
                                tstage = self.c+[1.0])

        # Storage for register banks and current index
        self._init_reg_banks()

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

        # Event handlers for advance_to
        self.completed_step_handlers = proxylist(self._get_plugins())

        # Sanity checks
        if self._controller_needs_errest and not self._stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        # Ensure the system is compatible with our formulation
        if 'std' not in systemcls.elementscls.formulations:
            raise RuntimeError(
                'System {0} does not support time stepping formulation std'
                .format(systemcls.name)
            )

    @property
    def soln(self):
        # If we do not have the solution cached then fetch it
        if not self._curr_soln:
            self._curr_soln = self.system.ele_scal_upts(self._idxcurr)
        
        # fix the possible nans in blanked cells
        if self.system.overset is True:
            iblank_cell = self.system.oset.griddata['iblank_cell']
            # consider multiple type of elements here
            etypeoff = [0]
            for eshape in self.system.ele_shapes[:-1]:
                neles = eshape[2]
                etypeoff.append(neles)

            for eoff, sol in zip(etypeoff, self._curr_soln):
                for idx in range(sol.shape[2]):
                    cidx = eoff + idx
                    if iblank_cell[cidx] != 1: # blanked
                        sol[:,0,idx] = 1.0
                        sol[:,1,idx] = 0.0
                        sol[:,2,idx] = 0.0
                        sol[:,3,idx] = 0.0
                        sol[:,4,idx] = 10.0
                    else:
                        pass
        return self._curr_soln

    @property
    def _controller_needs_errest(self):
        pass

    @property
    def _stepper_has_errest(self):
        pass
