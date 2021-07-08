# -*- coding: utf-8 -*-
from pyfr.solvers.baseadvec import BaseAdvectionSystem
import numpy as np

# for debugging purpose
from convert import *

class BaseAdvectionDiffusionSystem(BaseAdvectionSystem):
    def rhs(self, t, uinbank, foutbank):
        runall = self.backend.runall
        q1, q2 = self._queues
        kernels = self._kernels

        self._bc_inters.prepare(t)

        self.eles_scal_upts_inb.active = uinbank
        self.eles_scal_upts_outb.active = foutbank

        tstage = np.array([self.tcurr+i*self.dtcurr for i in self.tstage])
        istage = next(i for i, _ in enumerate(tstage) if np.isclose(_, t, 1e-4*self.dtcurr))
        fpdtype = self.backend.fpdtype
        
        # do the overset related stuff first
        if self.mvgrid and self.overset:
            # check t 
            tn, tn1 = self.tcurr, self.tcurr+self.dtcurr
            # is first stage
            if istage == 0:
                # move grid to stage end
                # first move faces using PyFR kernel
                
                fpdtype = self.backend.fpdtype
                motion = self._calc_motion(tn1, tn1, self.motioninfo, fpdtype)
                of = motion['offset']
                R = motion['Rmat']

                self.oset.unblankPart1( motion )
                
                # move grid to stage start
                # first move faces using PyFR kernel
                motion = self._calc_motion(tn, tn, self.motioninfo, fpdtype)
                of = motion['offset']
                R = motion['Rmat']
                
                # these are used by get_face_nodes get_cell_nodes
                q1 << kernels['eles','updateplocface'](
                    t=t, 
                    r00 = R[0],  r01 = R[1],  r02 = R[2],
                    r10 = R[3],  r11 = R[4],  r12 = R[5],
                    r20 = R[6],  r21 = R[7],  r22 = R[8],
                    ofx = of[0], ofy = of[1], ofz = of[2]
                )

                q1 << kernels['eles','updateploc'](
                    t=t, 
                    r00 = R[0],  r01 = R[1],  r02 = R[2],
                    r10 = R[3],  r11 = R[4],  r12 = R[5],
                    r20 = R[6],  r21 = R[7],  r22 = R[8],
                    ofx = of[0], ofy = of[1], ofz = of[2]
                )
                
                runall([q1])
                # is in  different stream
                # need to sync before proceed
                self.oset.sync_device()
                #self.oset.update_transform(motion['Rmat'], motion['offset'])
                self.oset.unblankPart2( motion )
                self.oset.performPointConnectivity()
            else:
                # for other stages once new blanking is setup
                # just need to move the interfaces  for pointconnectivity
                dt = tstage[istage] - tstage[istage-1]
                motion = self._calc_motion(t, t, self.motioninfo, fpdtype)
                of = motion['offset']
                R = motion['Rmat']
                
                #self.oset.move_solvercoords(motion)
                # update face grids
                q1 << kernels['eles','updateplocface'](
                    t=t, 
                    r00 = R[0],  r01 = R[1],  r02 = R[2],
                    r10 = R[3],  r11 = R[4],  r12 = R[5],
                    r20 = R[6],  r21 = R[7],  r22 = R[8],
                    ofx = of[0], ofy = of[1], ofz = of[2]
                )

                q1 << kernels['eles','updateploc'](
                    t=t, 
                    r00 = R[0],  r01 = R[1],  r02 = R[2],
                    r10 = R[3],  r11 = R[4],  r12 = R[5],
                    r20 = R[6],  r21 = R[7],  r22 = R[8],
                    ofx = of[0], ofy = of[1], ofz = of[2]
                )
                runall([q1])
                # is in  different stream
                # need to sync before proceed
                self.oset.sync_device()
                # also need to move the body grid
                # eventally move to last position
                self.oset.update_transform(motion['Rmat'], motion['offset'])
                self.oset.update_adt_transform( motion )
                self.oset.move_flat( motion )
                self.oset.move_nested( motion )
                self.oset.move_on_cpu()
                
                self.oset.performPointConnectivity()

        # here update the solution on artbnd
        # test
        #    self.oset.exchangeSolution()
        # run moving grid related kernels first
        if self.mvgrid is True:
            #if istage == 0:
            #    dt = 0.0
            #else:
            #    dt = tstage[istage] - tstage[istage-1]
            motion = self._calc_motion(t, t, self.motioninfo, fpdtype)
            
            of = motion['offset']
            R = motion['Rmat']

            if self.overset is False:
                q1 << kernels['eles','updateplocface'](
                    t=t, 
                    r00 = R[0],  r01 = R[1],  r02 = R[2],
                    r10 = R[3],  r11 = R[4],  r12 = R[5],
                    r20 = R[6],  r21 = R[7],  r22 = R[8],
                    ofx = of[0], ofy = of[1], ofz = of[2]
                )
                #runall([q1])

                q1 << kernels['eles','updateploc'](
                    t=t, 
                    r00 = R[0],  r01 = R[1],  r02 = R[2],
                    r10 = R[3],  r11 = R[4],  r12 = R[5],
                    r20 = R[6],  r21 = R[7],  r22 = R[8],
                    ofx = of[0], ofy = of[1], ofz = of[2]
                )

                #runall([q1])
                
            omg = motion['omega']
            pivot = motion['pivot']
            tvel = motion['tvel']

            q1 << kernels['eles','updatemvel'](
              t=t,
              omegax = omg[0],   omegay = omg[1],   omegaz = omg[2],
              pivotx = pivot[0]+of[0], pivoty = pivot[1]+of[1], pivotz = pivot[2]+of[2],
              tvelx = tvel[0],   tvely = tvel[1],   tvelz = tvel[2]
            )
            
            #runall([q1])
            # directly calculate
            q1 << kernels['eles','updatemvelface'](
              t=t,
              omegax = omg[0],   omegay = omg[1],   omegaz = omg[2],
              pivotx = pivot[0]+of[0], pivoty = pivot[1]+of[1], pivotz = pivot[2]+of[2],
              tvelx = tvel[0],   tvely = tvel[1],   tvelz = tvel[2]
            )
            #runall([q1])

            q1 << kernels['eles','updatejacobimatrix']()
            q1 << kernels['eles','updatejaco']()

            #runall([q1])

            # interpolate to interfaces
            #q1 << kernels['eles','mvel_fpts_int']()
            q1 << kernels['eles','smat_fpts_int']()
            #q1 << kernels['eles','mvel_fpts_ext']()
            q1 << kernels['eles','smat_fpts_ext']()
            #runall([q1])
            # 
            q1 << kernels['iint','updatepnorm']()
            q1 << kernels['mpiint','updatepnorm']()
            q1 << kernels['bcint','updatepnorm']()
            #runall([q1])
        
        q1 << kernels['eles', 'disu_ext']()

        # in case there is overset boudary
        if ('mpiint','scal_fpts_pack') in kernels:
            q1 << kernels['mpiint', 'scal_fpts_pack']()
        #if self.mvgrid is True:
        if ('mpiint','vect_fpts_mvel_pack') in kernels:
            #q1 << kernels['eles','mvel_fpts_ext']()
            q1 << kernels['mpiint','vect_fpts_mvel_pack']()
        runall([q1])


        q1 << kernels['eles', 'disu_int']()
        runall([q1])
        
        # here update the solution on artbnd
        if self.mvgrid and self.overset:
            self.oset.exchangeSolution()
            self.oset.sync_device()

        if ('eles', 'copy_soln') in kernels:
            q1 << kernels['eles', 'copy_soln']()
        if ('iint', 'copy_fpts') in kernels:
            q1 << kernels['iint', 'copy_fpts']()

        q1 << kernels['iint', 'con_u']()
        q1 << kernels['bcint', 'con_u'](t=t)
        if ('eles', 'shocksensor') in kernels:
            q1 << kernels['eles', 'shocksensor']()
            q1 << kernels['mpiint', 'artvisc_fpts_pack']()

        # gradient
        q1 << kernels['eles', 'tgradpcoru_upts']()
        
        # in case there is overset boundary
        if ('mpiint','scal_fpts_send') in kernels:
            q2 << kernels['mpiint', 'scal_fpts_send']()
            q2 << kernels['mpiint', 'scal_fpts_recv']()
            q2 << kernels['mpiint', 'scal_fpts_unpack']()

        # send, recv and unpack state and mvel
        #if self.mvgrid is True:
        if ('mpiint','vect_fpts_mvel_send') in kernels:
            q2 << kernels['mpiint', 'vect_fpts_mvel_send']()
            q2 << kernels['mpiint', 'vect_fpts_mvel_recv']()
            q2 << kernels['mpiint', 'vect_fpts_mvel_unpack']()

        runall([q1, q2])


        q1 << kernels['mpiint', 'con_u']()
        q1 << kernels['eles', 'tgradcoru_upts_ext']()
        q1 << kernels['eles', 'gradcoru_upts_ext']()
        q1 << kernels['eles', 'gradcoru_fpts_ext']()
        
        if ('mpiint','vect_fpts_pack') in kernels:
            q1 << kernels['mpiint', 'vect_fpts_pack']()

        if ('eles', 'shockvar') in kernels:
            if('mpiint','artvisc_fpts_send') in kernels:
                q2 << kernels['mpiint', 'artvisc_fpts_send']()
                q2 << kernels['mpiint', 'artvisc_fpts_recv']()
                q2 << kernels['mpiint', 'artvisc_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['eles', 'tgradcoru_upts_int']()
        runall([q1])


        q1 << kernels['eles', 'gradcoru_upts_int']()
        q1 << kernels['eles', 'gradcoru_fpts_int']()
        if ('eles', 'gradcoru_qpts') in kernels:
            q1 << kernels['eles', 'gradcoru_qpts']()

        runall([q1])
        
        q1 << kernels['eles', 'tdisf']()
        runall([q1])
        # first correction kernel
        q1 << kernels['eles', 'tdivtpcorf']()

        runall([q1])

        # avoid overwrite
        if self.mvgrid and self.overset:
            self.oset.exchangeGradient()
            self.oset.sync_device()

        q1 << kernels['iint', 'comm_flux']()
        q1 << kernels['bcint', 'comm_flux'](t=t)
        
        if ('mpiint','vect_fpts_send') in kernels:
            q2 << kernels['mpiint', 'vect_fpts_send']()
            q2 << kernels['mpiint', 'vect_fpts_recv']()
            q2 << kernels['mpiint', 'vect_fpts_unpack']()

        runall([q1, q2])

        q1 << kernels['mpiint', 'comm_flux']()
        # second correction kernel
        q1 << kernels['eles', 'tdivtconf']()
        
        runall([q1])
        # caculate after flux calculations are done
        if self.mvgrid is True:
            q1 << kernels['eles', 'tgcl']()
            q1 << kernels['eles', 'tdivgcl']()
            q1 << kernels['eles', 'tdivcorgcl']()
            runall([q1])

        if ('eles', 'tdivf_qpts') in kernels:
            q1 << kernels['eles', 'tdivf_qpts']()
            q1 << kernels['eles', 'negdivconf'](t=t)
            q1 << kernels['eles', 'divf_upts']()
        else:
            q1 << kernels['eles', 'negdivconf'](t=t)

        runall([q1])
