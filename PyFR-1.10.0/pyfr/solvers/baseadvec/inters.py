# -*- coding: utf-8 -*-

import math

from pyfr.solvers.base import BaseInters, get_opt_view_perm
from pyfr.nputil import npeval


class BaseAdvectionIntInters(BaseInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, elemap, cfg)
        
        # record information for overset
        self.lhs = lhs
        self.rhs = rhs

        const_mat = self._const_mat
        ncons_mat = self._non_const_mat
        # Compute the `optimal' permutation for our interface
        self._gen_perm(lhs, rhs)

        # Generate the left and right hand side view matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = self._scal_view(rhs, 'get_scal_fpts_for_inter')
        
        # Generate LHS and RHS view matrices for coordinates related
        # i.e. mvel and smat
        self._vect_lhs_mvel = self._vect_rhs_mvel = None
        self._vect_lhs_smat = self._vect_rhs_smat = None
        self._scal_lhs_mvel = self._scal_rhs_mvel = None

        
        if self.mvgrid is True: 
            # lhs and rhs be consistent even if the matrix 
            # dims are different from previous ones
            # get view matrix for mvel and smat
            self.mvgrid_cls.update_int_inters_view_matrix( self, lhs, rhs)
            # Allocate nonconsant matrices with initial value
            self._mag_pnorm_lhs  = ncons_mat( lhs, 'get_mag_pnorms_for_inter')
            self._norm_pnorm_lhs = ncons_mat( lhs, 'get_norm_pnorms_for_inter')
            # snorm is the face normal in standard element
            self._snorm = const_mat(lhs, 'get_snorms_for_inter')
            
            tplargs = dict(ndims=self.ndims,nvars=self.nvars)
            # Kernel to update pnorm with interpolated smat
            self.mvgrid_cls.update_face_norm_backend(
                self,self.kernels,tplargs,[self.ninterfpts]
            )
        else:
            # Generate the constant matrices
            self._mag_pnorm_lhs  = const_mat( lhs, 'get_mag_pnorms_for_inter')
            self._norm_pnorm_lhs = const_mat( lhs, 'get_norm_pnorms_for_inter')

    def _gen_perm(self, lhs, rhs):
        # Arbitrarily, take the permutation which results in an optimal
        # memory access pattern for the LHS of the interface
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter',
                                       self.elemap)


class BaseAdvectionMPIInters(BaseInters):
    # Tag used for MPI
    MPI_TAG = 2314

    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        # need to distinguish overset from regular mpi interfaces
        # check if rhsrank is None or not
        super().__init__(be, lhs, elemap, cfg)
        self._rhsrank = rhsrank
        self._rallocs = rallocs
        
        self.lhs = lhs

        const_mat = self._const_mat
        ncons_mat = self._non_const_mat

        # Generate the left hand view matrix and its dual
        self._scal_lhs = self._scal_xchg_view(lhs, 'get_scal_fpts_for_inter')
        self._scal_rhs = be.xchg_matrix_for_view(self._scal_lhs)

        self._vect_lhs_mvel = self._vect_rhs_mvel = None
        self._vect_lhs_smat = self._vect_rhs_smat = None
        self._scal_lhs_mvel = self._scal_rhs_mvel = None

        # get the view of the artbnd storage
        if self.overset is True:
            self._scal_lhs_artbnd = self._scal_xchg_view_artbnd(lhs, 'get_scal_fpts_artbnd_for_inter')
            self._scal_rhs_artbnd = self._be._scal_xchg_view_artbnd(self._scal_lhs_artbnd)
        else:
            self._scal_lhs_artbnd = None
            self._scal_rhs_artbnd = None

        if self.mvgrid is True:
            self.mvgrid_cls.update_mpi_inters_view_matrix(self, lhs)

            self._mag_pnorm_lhs  = ncons_mat( lhs, 'get_mag_pnorms_for_inter')
            self._norm_pnorm_lhs = ncons_mat( lhs, 'get_norm_pnorms_for_inter')
            self._snorm = const_mat(lhs, 'get_snorms_for_inter')

            tplargs = dict(ndims=self.ndims,nvars=self.nvars)
            # Kernel to update pnorm with interpolated smat
            self.mvgrid_cls.update_face_norm_backend(
                self,self.kernels,tplargs,[self.ninterfpts]
            )
        else:
            self._mag_pnorm_lhs  = const_mat( lhs, 'get_mag_pnorms_for_inter')
            self._norm_pnorm_lhs = const_mat( lhs, 'get_norm_pnorms_for_inter')

        # Kernels
        if self._rhsrank == None:
            # overset face does not need this
            if self.mvgrid is True:
                # need a kernel here to take care of mvelr
                # nope taken care of in mpicflux
                pass 
            pass
        else:
            self.kernels['scal_fpts_pack'] = lambda: be.kernel(
                'pack', self._scal_lhs
            ) 
            self.kernels['scal_fpts_send'] = lambda: be.kernel(
                'send_pack', self._scal_lhs, self._rhsrank, self.MPI_TAG
            ) 
            self.kernels['scal_fpts_recv'] = lambda: be.kernel(
                'recv_pack', self._scal_rhs, self._rhsrank, self.MPI_TAG
            ) 
            self.kernels['scal_fpts_unpack'] = lambda: be.kernel(
                'unpack', self._scal_rhs
            )  

            # Kernels to send grid velocity
            # 
            if self.mvgrid is True:
                self.kernels['vect_fpts_mvel_pack'] = lambda: be.kernel(
                    'pack', self._vect_lhs_mvel
                ) 
                self.kernels['vect_fpts_mvel_send'] = lambda: be.kernel(
                    'send_pack', self._vect_lhs_mvel, self._rhsrank, self.MPI_TAG
                ) 
                self.kernels['vect_fpts_mvel_recv'] = lambda: be.kernel(
                    'recv_pack', self._vect_rhs_mvel, self._rhsrank, self.MPI_TAG
                ) 
                self.kernels['vect_fpts_mvel_unpack']=lambda: be.kernel(
                    'unpack', self._vect_rhs_mvel
                ) 


class BaseAdvectionBCInters(BaseInters):
    type = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfg)
        self.cfgsect = cfgsect
        
        # record information for overset
        self.lhs = lhs

        const_mat = self._const_mat
        ncons_mat = self._non_const_mat

        # For BC interfaces, which only have an LHS state, we take the
        # permutation which results in an optimal memory access pattern
        # iterating over this state.
        self._perm = get_opt_view_perm(lhs, 'get_scal_fpts_for_inter', elemap)

        # LHS view and constant matrices
        self._scal_lhs = self._scal_view(lhs, 'get_scal_fpts_for_inter')

        self._vect_lhs_mvel =  None
        self._vect_lhs_smat =  None
        self._scal_lhs_mvel =  None

        if self.mvgrid is True:
            self.mvgrid_cls.update_bc_inters_view_matrix(self, lhs)

            self._mag_pnorm_lhs = ncons_mat( lhs, 'get_mag_pnorms_for_inter')
            self._norm_pnorm_lhs = ncons_mat( lhs, 'get_norm_pnorms_for_inter')
            self._snorm = const_mat(lhs, 'get_snorms_for_inter')
            # for debugging purpose
            #self._mag_pnorm_lhs_deg  = const_mat( lhs, 'get_mag_pnorms_for_inter')
            #self._norm_pnorm_lhs_deg = const_mat( lhs, 'get_norm_pnorms_for_inter')
            tplargs = dict(ndims=self.ndims,nvars=self.nvars)
            # Kernel to update pnorm with interpolated smat
            self.mvgrid_cls.update_face_norm_backend(
                self,self.kernels,tplargs,[self.ninterfpts]
            )
        else:
            self._mag_pnorm_lhs = const_mat( lhs, 'get_mag_pnorms_for_inter')
            self._norm_pnorm_lhs = const_mat( lhs, 'get_norm_pnorms_for_inter')

        # Make the simulation time available inside kernels
        self._set_external('t', 'scalar fpdtype_t')

    def _eval_opts(self, opts, default=None):
        # Boundary conditions, much like initial conditions, can be
        # parameterized by values in [constants] so we must bring these
        # into scope when evaluating the boundary conditions
        cc = self.cfg.items_as('constants', float)

        cfg, sect = self.cfg, self.cfgsect

        # Evaluate any BC specific arguments from the config file
        if default is not None:
            return [npeval(cfg.getexpr(sect, k, default), cc) for k in opts]
        else:
            return [npeval(cfg.getexpr(sect, k), cc) for k in opts]

    def _exp_opts(self, opts, lhs, default={}):
        cfg, sect = self.cfg, self.cfgsect

        subs = cfg.items('constants')
        subs.update(x='ploc[0]', y='ploc[1]', z='ploc[2]')
        subs.update(abs='fabs', pi=str(math.pi))

        exprs = {}
        for k in opts:
            if k in default:
                exprs[k] = cfg.getexpr(sect, k, default[k], subs=subs)
            else:
                exprs[k] = cfg.getexpr(sect, k, subs=subs)

        if (any('ploc' in ex for ex in exprs.values()) and
            'ploc' not in self._external_args):
            spec = 'in fpdtype_t[{0}]'.format(self.ndims)
            value = self._const_mat(lhs, 'get_ploc_for_inter')

            self._set_external('ploc', spec, value=value)

        return exprs
