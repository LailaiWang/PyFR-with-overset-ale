# -*- coding: utf-8 -*-
from collections import defaultdict, OrderedDict
import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where
from pyfr.solvers.moving_grid.movgrid import motion_exprs, calc_motion
from pyfr.quadrules import get_quadrule

# use some parallelism here 
from multiprocessing import Pool
from scipy.spatial import KDTree

# this is really time comsuming, let's use mp to accelerate since
# import multiprocessing

'''
This is a class to interpolate the solution from overset grid 
to background grid for quantatitive analysis

case 1: Isentropic Vortox Propagation
case 2: Taylor Green 
case 3: Turbulence Decaying
'''


class PostOverset(object):
    def __init__(self, elementscls, mesh, soln, mesh_inf, soln_inf, cfg, stats,blanking):
        self.elementscls = elementscls
        self.mesh = mesh
        self.soln = soln
        self.mesh_inf = mesh_inf
        self.soln_inf = soln_inf
        self.cfg = cfg
        self.stats = stats
        self.precision = self.cfg.get('backend','precision')
        self.fpdtype = np.float32 if self.precision == 'single' else np.float64
        self.t = self.fpdtype(self.stats.get('solver-time-integrator','tcurr'))
        
        # build up the motion information
        self.motion = defaultdict(list)
        # avoid repeatly compute this part       
        for gid in range(len(self.mesh)):
            motioninfo = motion_exprs(self.cfg,gid)
            self.motion[f'grid_{gid}'] = calc_motion(self.t, self.t, motioninfo, self.fpdtype)
        if not blanking:
            self.bksoln = self._merge_overset()

    def _merge_overset(self):
        ngparts = [0]
        
        # collect all the blanked upts in the backgroud mesh
        self.unblanked_overset      = defaultdict(list) # solution points
        self.unblanked_overset_shp  = defaultdict(list)
        self.unblanked_overset_eles = defaultdict(list) # mesh points

        for gid, amesh in enumerate(self.mesh):
            # extract the solution info from self.soln_inf
            soln_inf = OrderedDict((k, v) for k,v in self.soln_inf.items() 
                              if 'g{}'.format(gid) in k)
        
            ngparts.append(len(soln_inf))
            # build the offset 
            offset = sum(ngparts[:gid+1])

            if gid == 0:
                # the first grid is always the background grid
                # therefore, I need to first collect those blanked points first
                self._collect_blanked_upts(
                    amesh, self.mesh_inf[gid], self.soln, soln_inf, gid, offset
                )
            else:
                # other meshes are all overset meshes
                self._collect_upts_pool(
                    amesh, self.mesh_inf[gid], self.soln, soln_inf, gid, offset
                )
        
        # build up a tree use tree to search for solution
        # to save computational cost for post-processing
        self._build_trees()

        solnbackground = defaultdict(list)
        
        # loop over blaned elements, here are the solution points
        for part, eles in self.blanked_pts.items():
           
            blankedeidx = self.blanked_eid[part]

            partsoln = self.soln[part]
            
            # loop over elements
            for idxe in range(eles.shape[2]):
                #loop over solution points
                for idxspts in range(eles.shape[0]):
                    pts      = eles[idxspts, :, idxe]

                    # searching for the point in the forests
                    rst, destination = self._search_in_forests(pts)
                    
                    # destination stores the (uidx, eidx, partname)
                    _, (_,eidx), ovpart = destination

                    # do the interpolation
                    intp_soln = self._interpolation(
                        rst, self.unblanked_overset_eles[ovpart], eidx, ovpart
                    )
                    
                    bkeidx = blankedeidx[idxe]
                
                    partsoln[idxspts,:,bkeidx] = intp_soln[0,:]

            solnbackground[part] = partsoln
        
        return solnbackground

    def _interpolation(self, rst, eles, eidovset, part):
        '''
        subroutine to interpolate the nodal values
        using local coordinates
        '''
        eop = eles.basis.ubasis.nodal_basis_at(rst)
        esoln = self.soln[part][:,:,eidovset]
        
        intp_soln = eop @ esoln
        return intp_soln
    
    def _build_trees(self):
        '''
        Build up trees using coordinates
        First input:  the coordinates as dictionary
        Second input: shape functions
        Third input: the element class of each pool
        '''
        forests = defaultdict(list)
        
        for (name, eshape) in self.unblanked_overset.items():
            '''
            eshape (solution points) has a dimension as (nupts, dim, neles)
            need to update the coordinates first from the reference position
            to position at current time instance
            '''
            # infer the grid id from name
            gid = int(name.split('_')[1].split('-')[1][1:])

            # motioninfo = motion_exprs(self.cfg, gid)
            # need provide data type to this function
            # motion = calc_motion(self.t, self.t, motioninfo, self.fpdtype)

            motion = self.motion[f'grid_{gid}']

            '''
            I am sketching the algorithm here since I tend to forget it very soon
            
            Rmat is a 3x3 matrix
            Pivot is a 1x3 vector

            use np.tile to stack a pivot matrix
            
            let x0 be the reference coordinates xof be to offset, xp0 be reference pivot

            new coords after offset xn0: x0+xof 
            new pivot after pffset xnp: xp0+xof

            new coords: x
            (x-xnp) = R(xn0-xnp)
            x = R(xn0-xnp) + xnp

            '''
            
            # rotation matrix
            rmat   = np.array(motion['Rmat']).reshape(-1,3).astype(self.fpdtype)
            offset = np.array(motion['offset']).astype(self.fpdtype)
            pivot  = np.array(motion['pivot']).astype(self.fpdtype)
            
            # double check if I need to do this
            # offsetv = np.tile(offset, (eshape.shape[0], eshape.shape[2], 1))
            # pivotv  = np.tile(pivot,  (eshape.shape[0], eshape.shape[2], 1))
            
            # swap axes such that the dim is (nupts, neles, ndim)
            eshape = eshape.swapaxes(1,2)
            # add offset to eshape (original coordiantes)
            # eshape = eshape + offsetv
            eshape = eshape + offset
            # add offset to pivot (original coordiantes)
            # pivotv = pivotv + offsetv
            pivot  = pivot + offset
            
            # use einsum broadcast
            # eshape = np.einsum('ij,...j->...i', rmat, eshape-pivotv) + pivotv
            eshape = np.einsum('ij,...j->...i', rmat, eshape-pivot) + pivot

            # flatten the coords to buildup KD tree
            feupts = eshape.reshape(-1,eshape.shape[2])

            forests[name] = KDTree(feupts) 
        
        # save the forest of trees
        self.forests = forests

    def _search_in_forests(self, pts):
        '''
        An function to searhc a local portion of the mesh
        '''
        # ---------------------------------------------------------
        # 1. Search for nearest nb in trees
        # 2. identify the cell that contains the node
        # 3. evaluate the local coordinates for future interpolation
        # 4. This works for benchmarks with regular Cart. meshes
        #-----------------------------------------------------------
        
        eupts = [ (eles.shape[0], eles.shape[2]) for _, eles in self.unblanked_overset.items()]
    
        # check the node in the forest
        dmins, amins = zip(*[ tree.query(pts) for _, tree in self.forests.items()])
        
        # unravel the index 
        amins = [np.unravel_index(i, e) for i, e in zip(amins, eupts)]
        
        # part name
        partnames = [name for name in self.forests.keys()]

        # Reduce across different trees
        destination = min(zip(dmins, amins, partnames))
        # use this information get the accurate reference coordinates
        passflag, rst = self._eval_loc_coords(pts, destination) 
        
        if passflag == False:
            raise RuntimeError('node not find, which should not happen')

        # infer the element index
        return rst, destination

    def _prepare_eles(self, mesh, name, gid):
        '''
        Instantiate elementscls for future use sucn as intepolation, integration,
        Newton's method etc
        '''
        basiscls = subclass_where(BaseShape, name = name)
        eles = self.elementscls(basiscls, mesh, self.cfg, gid)
        return eles

    def _collect_blanked_upts(self, mesh, mesh_inf, soln, soln_inf, gid, goffset):
        # collect all the blanked upts in the backgroud mesh
        parts = defaultdict(list)
        for sk, (etype, shape) in soln_inf.items():
            part = sk.split('_')[-1]
        
            # mesh part name does not include g
            partoff = 'p{}'.format(int(part[1:])-goffset)
            etypem = etype.split('-g{}'.format(gid))[0]
            parts[part].append((f'spt_{etypem}_{partoff}', sk))
        
        # blanked points
        blanked_pts = defaultdict(list)
        blanked_eid = defaultdict(list)
        background_mesh = defaultdict(list)
        background_eles = defaultdict(list)
        
        for pfn, misil in parts.items(): # different element types
            for mk, sk in misil:
                name = mesh_inf[mk][0]
                mesh_0 = mesh[mk]
              #  soln = self.soln[sk].swapaxes(0, 1)
            
                skpre, skpost = sk.rsplit('_', 1)
                unblanked = self.soln[f'{skpre}_blank_{skpost}']
                # collect all blanked elements
                idxblanked = [idx for idx, stat in enumerate(unblanked) if stat == 0]

                # first instantiate the element class
                eles = self._prepare_eles(mesh_0, name, gid)
                # build the coordinates of upts of the elements
                elesupts = eles.ploc_at_np('upts')
                eles_blanked = elesupts[:,:,idxblanked]
                
                blanked_pts[sk] = eles_blanked
                blanked_eid[sk] = idxblanked
                background_mesh[sk] = elesupts
                background_eles[sk] = eles
        
        # pts to be searched
        self.blanked_pts = blanked_pts
        self.blanked_eid = blanked_eid
        self.background_esh = background_mesh 
        self.background_eles = background_eles

    def _collect_upts_pool(self, mesh, mesh_inf, soln, soln_inf, gid, goffset):

        parts = defaultdict(list)
        for sk, (etype, shape) in soln_inf.items():
            part = sk.split('_')[-1]
        
            # mesh part name does not include g
            partoff = 'p{}'.format(int(part[1:])-goffset)
            etypem = etype.split('-g{}'.format(gid))[0]
            parts[part].append((f'spt_{etypem}_{partoff}', sk))
        
        for pfn, misil in parts.items():
            for mk, sk in misil:
                name = mesh_inf[mk][0]
                mesh_0 = mesh[mk]
                #soln = self.soln[sk].swapaxes(0, 1)
            
                skpre, skpost = sk.rsplit('_', 1)
                unblanked = self.soln[f'{skpre}_blank_{skpost}']
                # collect all blanked elements
                idxunblanked = [idx for idx, stat in enumerate(unblanked) if stat == 1]

                # first instantiate the element class
                eles = self._prepare_eles(mesh_0, name, gid)
                # build the coordinates of upts of the elements
                elesupts = eles.ploc_at_np('upts')
                eles_unblanked = elesupts[:,:,idxunblanked]
                eles_m_unblanked = eles.eles[:,idxunblanked, :] # mesh points
                
                self.unblanked_overset[sk] = eles_unblanked
                self.unblanked_overset_shp[sk] = eles_m_unblanked
                self.unblanked_overset_eles[sk] = eles
    
    def _eval_loc_coords(self, xyz, destination):
        '''
        Evaluate the local coordinates using Newton's method
        The provides shape is the shape pionts aka. mesh nodes at reference position
        Need to move the reference position to the physical position at current time step
        '''
        ndims = 3
        
        # grab necessary information
        _, (_, eidx), partname = destination
        eles = self.unblanked_overset_eles[partname]
        eshape = self.unblanked_overset_shp[partname][:,eidx,:]
        # infer the grid id from name
        gid = int(partname.split('_')[1].split('-')[1][1:])
        
        # first move the reference coordinates
        #motioninfo = motion_exprs(self.cfg, gid)
        #motion = calc_motion(self.t, self.t, motioninfo, self.fpdtype)
        motion = self.motion[f'grid_{gid}']
        rmat   = np.array(motion['Rmat']).reshape(-1,3).astype(self.fpdtype)
        offset = np.array(motion['offset']).astype(self.fpdtype)
        pivot  = np.array(motion['pivot']).astype(self.fpdtype)
        
        # take advantage of broadcast in numpy
        eshape = eshape + offset
        pivot  = pivot + offset
        
        # this is the actual location of the shape of the elements
        eshape = np.einsum('ij,...j->...i', rmat, eshape-pivot) + pivot

        # using the real physical location to get the values
        stdshape = np.array(eles.basis.spts)
        stdlower = np.array([np.min(stdshape[:,i]) for i in range(ndims)])[None,...]
        stdupper = np.array([np.max(stdshape[:,i]) for i in range(ndims)])[None,...]
        
        coords = xyz[None, ...]
        rst_p = np.array([0.0,0.0,0.0])[None, ...]
        rst_n = rst_p
        # using rst to get the initial xyz
        eop    = eles.basis.sbasis.nodal_basis_at(rst_p)

        dxyz = eop @ eshape - coords
        # apply newton's method
        iters = 0
        while np.linalg.norm(dxyz)>1e-6:
            # get the jac_nodal_basis
            eop    = eles.basis.sbasis.nodal_basis_at(rst_n)
            eop_df = eles.basis.sbasis.jac_nodal_basis_at(rst_n).swapaxes(1,2)

            rst_n = rst_n -  np.dot((eop @ eshape - coords),
                         np.linalg.inv((eop_df @ eshape).reshape(ndims,ndims)))

            dxyz = eop @ eshape - coords
            iters = iters + 1
            if iters > 10:
                break

        if np.linalg.norm(dxyz) <= 1e-6: # converged
            
            # need to check if the convergence value is withint the range
            # formulations for tet hex prism are different
            
            dd = (rst_n > stdlower - 1e-5)
            ee = (rst_n < stdupper + 1e-5)
            ff = np.logical_and(dd,ee)
            if np.all(ff):
                passflag = True
            else:
                passflag = False
        else:
            passflag = False

        return passflag, rst_n

    def _integrate_background(self, soln, mesh, eles):
        
        errorpart = []
        volpart = []
        kinepart = []
        for part in soln.keys():
            isoln, imesh, ieles = soln[part], mesh[part], eles[part]
            # need to get the quadrature rule
            
            #r = get_quadrule()
            rname = self.cfg.get('solver-elements-' + ieles.basis.name, 'soln-pts')

            rcpdjac = ieles.rcpdjac_at_np('upts')
            jac = 1.0/rcpdjac
            smat = ieles.smat_at_np('upts').transpose(2, 0, 1, 3)
            gradop = ieles.basis.m4

            r = get_quadrule(ieles.basis.name, rname, ieles.basis.nupts)

            ipsoln = np.array(self.elementscls.con_to_pri(isoln.swapaxes(0,1), self.cfg))
            
            # testing
            #ipsoln = self._test_grad(imesh, ipsoln.swapaxes(0,1))
            #gradsoln = gradop @ ipsoln.reshape(ieles.basis.nupts, -1)
            gradsoln = gradop @ ipsoln.swapaxes(0,1).reshape(ieles.basis.nupts, -1)

            gradsoln = gradsoln.reshape(3, ieles.basis.nupts, 5, -1)

            gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln)

            kinetic = self._eval_kinetic(gradsoln, ipsoln)
            kinetic = kinetic.swapaxes(0,1)

            # evaluate the gradient if needed
            # evaluate the expression to be integrated
            errorsoln = self._exact_soln(imesh, ipsoln.swapaxes(0,1))
            e2soln = errorsoln*errorsoln

            vol = np.einsum('i,ik->ik', r.wts,jac)
            volt = np.sum(vol)

            kvol = np.einsum('ik,ijk->ijk', vol, kinetic)
            kvolt = kvol.swapaxes(0,1).reshape(4,-1)
            kvolt = np.sum(kvolt,axis = 1)

            evol = np.einsum('ik,ijk->ijk', vol, e2soln)
            evolt = evol.swapaxes(0,1).reshape(5,-1)
            evolt = np.sum(evolt, axis = 1)

            volpart.append(volt)
            errorpart.append(evolt)
            kinepart.append(kvolt)


        kinepart = np.array(kinepart)
        errorpart = np.array(errorpart)
        volpart = np.array(volpart)
        errorpart = errorpart/np.sum(volpart)
        kinepart = kinepart/np.sum(volpart)
        kinepartlist = [kinepart[0,i] for i in range(4)]
        kinepartlist.append(self.t)
        kinepartlist = np.atleast_2d(np.array(kinepartlist))
        np.savetxt(f"tgv_data_{self.t:.4g}", kinepartlist)

    def _eval_kinetic(self, gradsoln, soln):
        mu = float(self.cfg.get('constants','mu'))
        
        kinetic = np.zeros((4,soln.shape[1], soln.shape[2]))

        for eidx in range(gradsoln.shape[3]):
            for pts in range(gradsoln.shape[2]):
                
                m = gradsoln[1:4,:,pts, eidx]

                p = soln[4,pts,eidx]

                ux = m[0,0]
                uy = m[0,1]
                uz = m[0,2]

                vx = m[1,0]
                vy = m[1,1]
                vz = m[1,2]

                wx = m[2,0]
                wy = m[2,1]
                wz = m[2,2]

                s11 = ux
                s22 = vy
                s33 = wz

                s12= 0.5*(uy+vx)
                s13= 0.5*(uz+wx)
                s23= 0.5*(vz+wy)

                sum1 = 2.0*mu*(s11*s11+s22*s22+s33*s33+2.0*s12*s12+2.0*s13*s13+2.0*s23*s23)
                sum2 = 0.0*(ux+vy+wz)*(ux+vy+wz)
                sum3 = -p*(ux+vy+wz) 
                sum4 = sum1+sum2+sum3

                kinetic[0,pts,eidx] = sum1
                kinetic[1,pts,eidx] = sum2
                kinetic[2,pts,eidx] = sum3
                kinetic[3,pts,eidx] = sum4

        return kinetic

    def _test_grad(self, mesh, soln):
        exactsoln = np.zeros(soln.shape)
        for eidx in range(mesh.shape[2]):
            for pidx in range(mesh.shape[0]):
                pts = mesh[pidx,:,eidx]
                x = pts[0]
                y = pts[1]
                z = pts[2]

                S = 13.5    
                M = 0.4     
                R = 1.5     
                gamma = 1.4
            
                f = ((1 - x*x - y*y )/(2*R*R))
                rho = np.power(1 - S*S*M*M*(gamma - 1)*np.exp(2*f)/(8*np.pi*np.pi), 1/(gamma - 1))
                u = S*y*np.exp(f)/(2*np.pi*R)
                v = 1 - S*x*np.exp(f)/(2*np.pi*R)
                w = 0
                p = 1/(gamma*M*M)*np.power(1 - S*S*M*M*(gamma - 1)*np.exp(2*f)/(8*np.pi*np.pi), gamma/(gamma - 1))
                exactsoln[pidx,0,eidx] = 1.0*x
                exactsoln[pidx,1,eidx] = 2.0*y
                exactsoln[pidx,2,eidx] = 3.0*z
                exactsoln[pidx,3,eidx] = 1.0*x
                exactsoln[pidx,4,eidx] = 1.0*x

        return exactsoln 
    


    def _exact_soln_isentropic_vortex(self, mesh, soln):
        '''
        This is the script to process isentropic vortex propagation
        for overset
        '''
        exactsoln = np.zeros(soln.shape)
        for eidx in range(mesh.shape[2]):
            for pidx in range(mesh.shape[0]):
                pts = mesh[pidx,:,eidx]
                x = pts[0]
                y = pts[1]
                z = pts[2]

                S = 13.5    
                M = 0.4     
                R = 1.5     
                gamma = 1.4
            
                f = ((1 - x*x - y*y )/(2*R*R))
                rho = np.power(1 - S*S*M*M*(gamma - 1)*np.exp(2*f)/(8*np.pi*np.pi), 1/(gamma - 1))
                u = S*y*np.exp(f)/(2*np.pi*R)
                v = 1 - S*x*np.exp(f)/(2*np.pi*R)
                w = 0
                p = 1/(gamma*M*M)*np.power(1 - S*S*M*M*(gamma - 1)*np.exp(2*f)/(8*np.pi*np.pi), gamma/(gamma - 1))
                exactsoln[pidx,0,eidx] = rho
                exactsoln[pidx,1,eidx] = u
                exactsoln[pidx,2,eidx] = v
                exactsoln[pidx,3,eidx] = w
                exactsoln[pidx,4,eidx] = p

        return exactsoln - soln
