# -*- coding: utf-8 -*-

from pyfr.solvers.base import BaseSystem


class BaseAdvectionSystem(BaseSystem):
    _nqueues = 2

    def rhs(self, t, uinbank, foutbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self._bc_inters.prepare(t)

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        # run moving grid related kernels first
        if self.mvgrid is True:
            q1 << kernels['eles','updateploc'](t=t)
            q1 << kernels['eles','updatemvel'](t=t)
            q1 << kernels['eles','updatejacobimatrix']()
            q1 << kernels['eles','updatejaco']()
            runall([q1])

            # update the face norms and mags
            q1 << kernels['eles','mvel_fpts_int']()
            q1 << kernels['eles','smat_fpts_int']()
            q1 << kernels['iint','updatepnorm']()
            # these two are not necessary?
            q1 << kernels['mpiint','updatepnorm']()
            q1 << kernels['bcint','updatepnorm']()
            runall([q1])

        q1 << kernels['eles', 'disu_ext']()
        q1 << kernels['mpiint', 'scal_fpts_pack']()
        if self.mvgrid is True:
            q1 << kernels['eles','mvel_fpts_ext']()
            q1 << kernels['mpiint','vect_fpts_mvel_pack']()
        runall([q1])

        q1 << kernels['eles', 'disu_int']()
        if ('eles', 'copy_soln') in kernels:
            q1 << kernels['eles', 'copy_soln']()
        q1 << kernels['eles', 'tdisf']()
        q1 << kernels['eles', 'tdivtpcorf']()
        q1 << kernels['iint', 'comm_flux']()
        q1 << kernels['bcint', 'comm_flux'](t=t)

        q2 << kernels['mpiint', 'scal_fpts_send']()
        q2 << kernels['mpiint', 'scal_fpts_recv']()
        q2 << kernels['mpiint', 'scal_fpts_unpack']()
        if self.mvgrid is True:
            q2 << kernels['mpiint', 'vect_fpts_mvel_send']()
            q2 << kernels['mpiint', 'vect_fpts_mvel_recv']()
            q2 << kernels['mpiint', 'vect_fpts_mvel_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'comm_flux']()
        q1 << kernels['eles', 'tdivtconf']()

        # caculate alfer flux calculations are done
        if self.mvgrid is True:
            q1 << kernels['eles', 'tgcl']()
            q1 << kernels['eles', 'tdivgcl']()

        if ('eles', 'tdivf_qpts') in kernels:
            q1 << kernels['eles', 'tdivf_qpts']()
            q1 << kernels['eles', 'negdivconf'](t=t)
            q1 << kernels['eles', 'divf_upts']()
        else:
            q1 << kernels['eles', 'negdivconf'](t=t)
        runall([q1])
