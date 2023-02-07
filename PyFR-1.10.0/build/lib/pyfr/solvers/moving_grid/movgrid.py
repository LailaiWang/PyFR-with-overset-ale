# -*- coding: utf-8 -*-
from pyfr.backends.base.kernels import ComputeMetaKernel
import sympy as sym
import numpy as np
from collections import OrderedDict
class MovingGrid(object):
    '''
    MovingGrid class does not hold any actual data 
    Only provide the functionality for ALE formulation
    '''
    def __init__(self, antialias, gid, cfg):
        self.mvelmap = {2:['mvx','mvy'], 3:['mvx','mvy','mvz']}
        self.antialias = antialias
        self.gid = gid
        self.cfg = cfg
    
    # for memory allocation
    def update_bufs(self, bufs):
        # grid velocity as a vector 1 row ndims col
        # gcl as a scalar 1 in both upts and fpts
        bufs |= {'vect_upts_mvel','scal_upts_gcl', 'scal_fpts_gcl'}
        
        # for testing
        #bufs |= {'vect_upts_smat'}

        if 'flux' in self.antialias or 'div-flux' in self.antialias:
            bufs |= {'vect_qpts_mvel'}
        
        bufs |= {'vect_fpts_mvel'}
        bufs |= {'vect_fpts_smat'}
        return bufs
    
    def allocate_bufs(self,elemcls,sbufs,vsmtalloc,vmelalloc,sgclalloc,snpts):
        if 'vect_upts_mvel' in sbufs:
            elemcls._vect_upts_mvel = vmelalloc('vect_upts_mvel',snpts['nupts'])
        if 'vect_qpts_mvel' in sbufs:
            elemcls._vect_qpts_mvel = vmelalloc('vect_qpts_mvel',snpts['nqpts'])
        if 'vect_fpts_mvel' in sbufs:
            elemcls._vect_fpts_mvel = vmelalloc('vect_qpts_mvel',snpts['nfpts'])
        if 'vect_fpts_smat' in sbufs:
            elemcls._vect_fpts_smat = vsmtalloc('vect_fpts_smat',snpts['nfpts'])
        if 'scal_upts_gcl'  in sbufs:
            elemcls._scal_upts_gcl  = sgclalloc('scal_upts_gcl', snpts['nupts'])
        if 'scal_fpts_gcl'  in sbufs:
            elemcls._scal_fpts_gcl  = sgclalloc('scal_fpts_gcl', snpts['nfpts'])
    
    def update_int_inters_view_matrix(self,intercls,lhs,rhs):
        intercls._vect_lhs_mvel = intercls._vect_view_mvel(
            lhs,'get_vect_fpts_mvel_for_inter'
        )

        intercls._vect_rhs_mvel = intercls._vect_view_mvel(
            rhs,'get_vect_fpts_mvel_for_inter'
        )

        intercls._vect_lhs_smat = intercls._vect_view_smat(
            lhs,'get_vect_fpts_smat_for_inter'
        )
        intercls._vect_rhs_smat = intercls._vect_view_smat(
            rhs,'get_vect_fpts_smat_for_inter'
        )
        intercls._scal_lhs_mvel = intercls._scal_view_mvel(
            lhs,'get_scal_fpts_mvel_for_inter'
        )
        intercls._scal_rhs_mvel = intercls._scal_view_mvel(
            rhs,'get_scal_fpts_mvel_for_inter'
        )
    
    def update_mpi_inters_view_matrix(self,intercls,lhs):
        intercls._vect_lhs_mvel = intercls._vect_xchg_view_mvel(
            lhs,'get_vect_fpts_mvel_for_inter'
        )
        intercls._vect_rhs_mvel = intercls._be.xchg_matrix_for_view(
            intercls._vect_lhs_mvel
        )
        intercls._vect_lhs_smat = intercls._vect_xchg_view_smat(
            lhs,'get_vect_fpts_smat_for_inter'
        )
        intercls._vect_rhs_smat = intercls._be.xchg_matrix_for_view(
            intercls._vect_lhs_smat
        )
        intercls._scal_lhs_mvel = intercls._scal_xchg_view_mvel(
            lhs, 'get_scal_fpts_mvel_for_inter'
        )
        intercls._scal_rhs_mvel = intercls._be.xchg_matrix_for_view(
            intercls._scal_lhs_mvel
        )

    def update_bc_inters_view_matrix(self, intercls, lhs):
        intercls._vect_lhs_mvel = intercls._vect_view_mvel(
            lhs,'get_vect_fpts_mvel_for_inter'
        )
        intercls._vect_lhs_smat = intercls._vect_view_smat(
            lhs,'get_vect_fpts_smat_for_inter'
        )
        intercls._scal_lhs_mvel = intercls._scal_view_mvel(
            lhs,'get_scal_fpts_mvel_for_inter'
        )
    
    # kernel to interpolate smat to fpts
    def interpolate_mvel_smat_kernel(self,elemcls,kernels,slicem):
        # see eg of interpolate gradient
        # interpolate grid velocity to fpts
        for s, neles in elemcls._ext_int_sides:
            #if fluxaa or (divfluxaa and solnsrc):
            if 'flux' in self.antialias or 'div-flux' in self.antialias:
                pass
            else:
                def mvel_fpts(s=s):
                    nupts, nfpts = elemcls.nupts, elemcls.nfpts
                    vupts, vfpts = elemcls._vect_upts_mvel, elemcls._vect_fpts_mvel
                    muls = [elemcls._be.kernel('mul', elemcls.opmat('M0'),
                                slicem(vupts, s, i*nupts, (i + 1)*nupts),
                                slicem(vfpts, s, i*nfpts, (i + 1)*nfpts))
                            for i in range(elemcls.ndims)]

                    return ComputeMetaKernel(muls)
                kernels['mvel_fpts_'+s] = mvel_fpts
        
        # interpolate smat to fpts
        for s,neles in elemcls._ext_int_sides:
            def smat_fpts(s=s):
                nupts, nfpts = elemcls.nupts, elemcls.nfpts
                vupts, vfpts = elemcls.smat_at_ncon('upts'),elemcls._vect_fpts_smat

                # Exploit the block-diagonal form of the operator
                muls = [elemcls._be.kernel('mul', elemcls.opmat('M0'),
                           slicem(vupts, s, i*nupts, (i + 1)*nupts),
                           slicem(vfpts, s, i*nfpts, (i + 1)*nfpts))
                        for i in range(elemcls.ndims)]

                return ComputeMetaKernel(muls)
            kernels['smat_fpts_'+s] = smat_fpts

    # use backend to update mesh
    # maybe not
    def update_mesh_backend():
        pass
    
    # update mesh using solution points
    def update_mesh_by_spts():
        pass

    # physical location of solution points 
    def update_ploc_backend(self,elemcls,kernels,tplargs,dims,ploc):
        # register kernel with backend
        elemcls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.upploc'
        )
        # eg: for CUDA, dims is for grid/thread dimension
        kernels['updateploc'] = lambda: elemcls._be.kernel(
            'upploc', tplargs=tplargs, dims=dims,
            ploc=ploc, pref = elemcls.ploc_at('upts')
        )

    def update_ploc_face_backend(self,elemcls,kernels,tplargs,dims,plocfpts):
        # register kernel with backend
        elemcls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.upplocface'
        )
        # eg: for CUDA, dims is for grid/thread dimension
        kernels['updateplocface'] = lambda: elemcls._be.kernel(
            'upplocface', tplargs=tplargs, dims=dims, 
            ploc=plocfpts, pref = elemcls.ploc_at('fpts')
        )
    
    # unnormalize face norm
    def update_face_norm_backend(self,intercls,kernels,tplargs,dims):
        intercls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.uppnorm'
        )

        kernels['updatepnorm'] = lambda: intercls._be.kernel(
            'uppnorm',tplargs=tplargs,dims = dims,
            snorm=intercls._snorm,smat=intercls._vect_lhs_smat, 
            magn=intercls._mag_pnorm_lhs,norm=intercls._norm_pnorm_lhs
        )

    # return a kernel to update jaco on backend
    # need corresponding operators 
    def update_jaco_backend(self,elemcls,kernels,tplargs,dims,
                            ploc,smat,smatref, djac, djacref):
        elemcls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.upjaco'
        )
        
        # first kernel is to use basic operation to get Jacobi
        # and store data in smat
        kernels['updatejacobimatrix'] = lambda: elemcls._be.kernel(
            'mul',elemcls.opmat('M4'),ploc,out=smat
        )

        # use data stored in smat to update covariance matrix
        kernels['updatejaco'] = lambda: elemcls._be.kernel(
            'upjaco', tplargs=tplargs,dims=dims,
            smats=smat, smatr = smatref, rcpdjac=djac, rcpdjacr = djacref
        )
    
    '''
    rotate the jacobian is not tested
    Do not use
    '''
    def rotate_jaco_backend(self,elemcls,kernels,tplargs,dims,
                            ploc,smat,smatref, djac, djacref):
        elemcls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.rotjaco'
        )
        
        # use data stored in smat to update covariance matrix
        kernels['rotatejaco'] = lambda: elemcls._be.kernel(
            'rotjaco', tplargs=tplargs,dims=dims,
            smats=smat, smatr = smatref, rcpdjac=djac, rcpdjacr = djacref
        )

    
    #return a kernel to update mvel on backend
    def update_mvel_backend(self,elemcls,kernels,tplargs,dims,ploc,mvel):
        elemcls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.upmvel'
        )

        kernels['updatemvel'] = lambda: elemcls._be.kernel(
            'upmvel', tplargs=tplargs,dims=dims,ploc=ploc,mvel=mvel,
            pref=elemcls.ploc_at('upts')
        )

    def update_mvel_face_backend(self,elemcls,kernels,tplargs,dims,plocfpts,mvelfpts):
        elemcls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.upmvelface'
        )

        kernels['updatemvelface'] = lambda: elemcls._be.kernel(
            'upmvelface', tplargs=tplargs,dims=dims,ploc=plocfpts,mvel=mvelfpts,
            pref=elemcls.ploc_at('fpts')
        )
    

    def update_gcl_component(self,elemcls,kernels,tplargs,dims,smats,mvel):
        '''
        kernel to calculate 
           |J|ξ_t = |J|(mvelx*ξ_x+mvely*ξ_y+mvelz*ξ_z)
           |J|η_t = |J|(mvelx*η_x+mvely*η_y+mvelz*η_z)
           |J|ζ_t = |J|(mvelx*ζ_x+mvely*ζ_y+mvelz*ζ_z)
           and store in mvel
        For output purpose one can recalculate mvel on host
        '''

        elemcls._be.pointwise.register(
            'pyfr.solvers.moving_grid.kernels.gclcomponent'
        )
        
        # reuse smats and mvel
        kernels['tgcl'] = lambda: elemcls._be.kernel(
            'gclcomponent',tplargs=tplargs,dims=dims,smats=smats,mvel=mvel
        )

        # kernel to calculate div(G)=(∂(|J|ξ_t)/∂ξ+∂(|J|η_t)/∂η+∂(|J|ζ_t)/∂ζ)
        # \bold(u)*div(G) is added in negdivconf kernel

        # first kernel where loc values on solution points and flux points functions
        kernels['tdivgcl'] = lambda: elemcls._be.kernel(
            'mul', elemcls.opmat('M1 - M3*M2'), elemcls._vect_upts_mvel,
            out=elemcls._scal_upts_gcl
        )
        # second kernel where common flux fucntions
        kernels['tdivcorgcl'] = lambda: elemcls._be.kernel(
            'mul', elemcls.opmat('M3'), elemcls._scal_fpts_gcl, 
            out=elemcls._scal_upts_gcl,
            beta=1.0
        )


def motion_exprs(cfg, gid):
    t = sym.symbols('t')
        
    translation = cfg.get('motion-translate-g{}'.format(gid), 'ison')
    translation = True if translation == 'yes' else False 
        
    if translation is True:
        tranx = cfg.get('motion-translate-g{}'.format(gid), 'x')
        trany = cfg.get('motion-translate-g{}'.format(gid), 'y')
        tranz = cfg.get('motion-translate-g{}'.format(gid), 'z')

        tranx, trany, tranz = eval(tranx), eval(trany), eval(tranz)

        tvelx, tvely, tvelz = sym.diff(tranx,t), sym.diff(trany,t), sym.diff(tranz,t)
    else:
        tranx, trany, tranz = 0.0, 0.0, 0.0
        tvelx, tvely, tvelz = 0.0, 0.0, 0.0
        
    rotation = cfg.get('motion-rotate-g{}'.format(gid),'ison')
    rotation = True if rotation == 'yes' else False
        
    if rotation is True:
        pivot = cfg.get('motion-rotate-g{}'.format(gid), 'pivot')
        axis  = cfg.get('motion-rotate-g{}'.format(gid), 'axis')
        rott  = cfg.get('motion-rotate-g{}'.format(gid), 'rott')
        
        pivot, axis, rott = eval(pivot), eval(axis), eval(rott)
        
        angvel = sym.diff(rott,t)
    else:
        pivot, axis, rott, angvel = 0.0, 0.0, 0.0, 0.0

    motioninfo = OrderedDict()
        
    motioninfo['translation'] = translation
    motioninfo['tranx'] = tranx
    motioninfo['trany'] = trany
    motioninfo['tranz'] = tranz
    motioninfo['tvelx'] = tvelx
    motioninfo['tvely'] = tvely
    motioninfo['tvelz'] = tvelz
        
    motioninfo['rotation'] = rotation
    motioninfo['pivot'] = pivot
    motioninfo['axis'] = axis
    motioninfo['rott'] = rott
    motioninfo['angvel'] = angvel
        
    return motioninfo
    
def calc_motion(time, dt, motioninfo, fpdtype, sign = 1.0):
    t = sym.symbols('t')
    
    offset = [0.0, 0.0, 0.0]

    offset[0] = fpdtype(motioninfo['tranx'].subs(t, time))
    offset[1] = fpdtype(motioninfo['trany'].subs(t, time))
    offset[2] = fpdtype(motioninfo['tranz'].subs(t, time))

    tvel = [0.0, 0.0, 0.0]
    
    tvel[0] = fpdtype(motioninfo['tvelx'].subs(t, time))
    tvel[1] = fpdtype(motioninfo['tvely'].subs(t, time))
    tvel[2] = fpdtype(motioninfo['tvelz'].subs(t, time))
    
    angn = motioninfo['rott'].subs(t,time)
    ango = motioninfo['rott'].subs(t,time-dt)

    alpha = fpdtype((angn - ango)/2.0)

    if sign == -1.0:
        alpha = -alpha

    axs = np.array(motioninfo['axis'])
    axs = axs/np.linalg.norm(axs)
    Q = [np.cos(alpha), np.sin(alpha)*axs[0], np.sin(alpha)*axs[1], np.sin(alpha)*axs[2]]

    Rmat   = quaternion_rotation_matrix(Q)

    angvel = fpdtype(motioninfo['angvel'].subs(t,time))
    
    omega = [angvel*i for i in axs]

    pivot = motioninfo['pivot']

    motion = OrderedDict()

    motion['offset'] = offset
    motion['tvel'] = tvel
    motion['Rmat'] = Rmat
    motion['omega'] = omega
    motion['pivot'] = pivot

    return motion

def quaternion_rotation_matrix(Q):
    # Extract the values from Q
    q0, q1, q2, q3 = Q

    r00 = 2.0 * (q0 * q0 + q1 * q1) - 1.0 
    r01 = 2.0 * (q1 * q2 - q0 * q3) 
    r02 = 2.0 * (q1 * q3 + q0 * q2)

    r10 = 2.0 * (q1 * q2 + q0 * q3) 
    r11 = 2.0 * (q0 * q0 + q2 * q2) - 1.0 
    r12 = 2.0 * (q2 * q3 - q0 * q1) 

    r20 = 2.0 * (q1 * q3 - q0 * q2) 
    r21 = 2.0 * (q2 * q3 + q0 * q1) 
    r22 = 2.0 * (q0 * q0 + q3 * q3) - 1.0

    rot_matrix = [r00,r01,r02,r10,r11,r12,r20,r21,r22]

    return rot_matrix
