# -*- coding: utf-8 -*-

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where


class BaseWriter(object):
    def __init__(self, args):
        from pyfr.solvers.base import BaseSystem

        # infer outf names names if addmsh for overset
        import os
        extn = os.path.splitext(args.outf)[1]
        outf = os.path.splitext(args.outf)[0]

        nmsh = 1 if args.addmsh == None else 1+len(args.addmsh)
        self.outf = ['{0}-grid-{1}{2}'.format(outf,idx,extn) for idx in range(nmsh)]
        
        meshf = [args.meshf] if args.addmsh == None else [args.meshf]+args.addmsh

        # create a list of meshes 
        self.mesh = [NativeReader(a) for a in meshf] 

        if args.nsolnf == 1:
            self.soln = NativeReader(args.solnf)
            # be compatible with output of previous version
            soln_muuid = self.soln['mesh_uuid']

            import numpy as np
            if isinstance(soln_muuid, np.ndarray) == False:
                soln_muuid = np.array([soln_muuid]).astype('|S36')      
            
            # Check solution and mesh are compatible
            if ([amesh['mesh_uuid'].encode() for amesh in self.mesh] 
                != soln_muuid.tolist()):
                raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, meshf))

            # Check if we have another soln file to calculate the fluctuations
            # the averaged field is the rho u v w p primitive variables
            if args.kte is True:
                self.avsoln = NativeReader(args.avsolnf)
                self.outf = 'fluctuations_{}'.format(self.outf)
                
        else:
            # averaging either solution files or fluctuation files
            solnbase = args.solnfbase.split('.')
            fidxs = [int(solnbase[0])+args.incrsolnf*i for i in range(args.nsolnf)]
            fidxstr = ['{:03}'.format(a) for a in fidxs]
            
            # file names for a series of files to be averaged
            self.solnfs = [args.solnf.replace(solnbase[0],a) for a in fidxstr]
            
            # sanity check make sure all files exists
            from os import path
            existence = [path.exists(filename) for filename in self.solnfs]

            if all(existence) == False:
                fidx = existence.index(False)
                raise RuntimeError('Soln {} does not exists'.format(self.solnfs[fidx]))

            # set first soln as default soln
            self.soln = NativeReader(self.solnfs[0])

        # Load the configuration and stats files
        self.cfg = Inifile(self.soln['config'])
        self.stats = Inifile(self.soln['stats'])

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'soln')

        # Get element types and array shapes
        self.mesh_inf = [a.array_info('spt') for a in self.mesh]
        # using first soln file when multiple soln files are read in
        self.soln_inf = self.soln.array_info(self.dataprefix)

        # Dimensions
        self.ndims = next(iter(self.mesh_inf[0].values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls

        
        # Sanity check of the averaged file
        if args.kte:
            avstats = Inifile(self.avsoln['stats'])
            avprefix = avstats.get('data', 'prefix', 'soln')
            if avprefix != 'tavg':
                raise RuntimeError('Prefix of averaged file is not tavg')

            # write the pyfrs format of fluctuations
            self.write_fluctuations(args)

        if args.nsolnf != 1:
            # doing average
            self.reynolds_averaging(args)

    def to_spherical(self):
        
        if self.ndims != 3:
            raise RuntimeError('to_spherical only works for 3D')

        # using coordinates (x,y,z) to get the spherical ()
        import numpy as np
        from pyfr.shapes import BaseShape
        from collections import defaultdict

        sphere_trans = defaultdict(list)
        # mesh is turned into a list instead of instance
        # original mesh does not have grid id information
        for gid, ameshinfo in enumerate(self.mesh_inf):
            for partname, (etype, v) in ameshinfo.items():
                # construction an instance of the relevant elements class
                # element type name
                basiscls = subclass_where(BaseShape, name = etype)

                eles = self.elementscls(
                    basiscls, self.mesh[gid][partname], self.cfg, gid
                )
                
                # evaluate the location of solution points
                # shape = [nupts dim eles]
                plocupts = eles.ploc_at_np('upts')
                
                sphcoords = np.zeros(plocupts.shape).astype(self.dtype)
                
                # using plocupts to evaluate r theta phi
                sphcoords[:, 0, :] = np.linalg.norm(plocupts,axis = 1)
                sphcoords[:, 1, :] = np.arctan2(plocupts[:, 1, :], plocupts[:, 0, :])
                sphcoords[:, 2, :] = np.arccos(
                    np.clip(plocupts[:, 2, :]/sphcoords[:, 0, :], -1.0, 1.0)
                )
        
                # add grid id into     
                partname = partname.replace(etype, etype+'-g{}'.format(gid))
                sphere_trans[partname] = sphcoords
        # return r theta phi
        self.sphere_trans = sphere_trans
    
    def rotation(self, rtp, uvw):
        
        import numpy as np
        R = np.zeros((3,3)).astype(self.type)
        
        r, theta, phi = rtp

        R[0][0] = np.sin(theta)*np.cos(phi) 
        R[0][1] = np.sin(theta)*np.sin(phi)
        R[0][2] = np.cos(theta)

        R[1][0] = np.cos(theta)*np.cos(phi)
        R[1][2] = np.cos(theta)*np.sin(phi)
        R[1][3] = -np.sin(theta)

        R[2][0] = - np.sin(phi)
        R[2][1] = np.cos(phi)
        R[2][2] = 0.0

        return np.dot(R, uvw)
    
    def reynolds_averaging(self, args):
        import numpy as np

        self.dtype = np.dtype(args.precision).type

        items = list(self.soln)
        partname = [a for a in items if isinstance(self.soln[a], np.ndarray)]
        metadata = [a for a in items if isinstance(self.soln[a], str)]
            
        from collections import defaultdict
        solndata = defaultdict(list)

        if args.spherical:
            self.to_spherical()
            
        for part in partname:
            if 'idxs' not in part:
                # first soln file
                v = self.soln[part]
                # convert vectors into spherical coordinates                   
                if self.dataprefix == 'prime':
                    # calculate the Reynolds stress
                    # if its navier-stokes equations
                    if self.systemscls.name == 'navier-stokes':
                        uup = (v[:,1,:]*v[:,1,:])[:,None,:]
                        vvp = (v[:,2,:]*v[:,2,:])[:,None,:]
                        wwp = (v[:,3,:]*v[:,3,:])[:,None,:]

                        uvp = (v[:,1,:]*v[:,2,:])[:,None,:]
                        uwp = (v[:,1,:]*v[:,3,:])[:,None,:]
                        vwp = (v[:,2,:]*v[:,3,:])[:,None,:]
                        rys = np.concatenate((uup,vvp,wwp,uvp,uwp,vwp), axis = 1)
                        v = np.concatenate((v,rys), axis=1)
                    else:
                        pass
                # need to perform density weighted averaged for compressible flows
                solndata[part] = v

        # load all the data and doing average
        for idx,solnf in enumerate(self.solnfs[1:]):
            # read the soln file one by one to save ram in case one file is 
            # excessively large
            isoln =  NativeReader(solnf)
            for part in partname:
                if 'idxs' not in part:
                    iv = isoln[part]
                    if self.dataprefix == 'prime':
                        # calculate the Reynolds stress
                        if self.systemscls.name == 'navier-stokes':
                            uup = (iv[:,1,:]*iv[:,1,:])[:,None,:]
                            vvp = (iv[:,2,:]*iv[:,2,:])[:,None,:]
                            wwp = (iv[:,3,:]*iv[:,3,:])[:,None,:]
                            uvp = (iv[:,1,:]*iv[:,2,:])[:,None,:]
                            uwp = (iv[:,1,:]*iv[:,3,:])[:,None,:]
                            vwp = (iv[:,2,:]*iv[:,3,:])[:,None,:]
                            rys = np.concatenate((uup, vvp, wwp, uvp, uwp, vwp),axis = 1)

                            iv = np.concatenate((iv,rys), axis=1)
                        else: 
                            pass
                    # here ends up with averaged rho averaged u v w p 
                    # or averaged rho u' v' w' p' and averaged reynolds stress
                    solndata[part] = (solndata[part]*(idx+1) + iv)/(idx+2)
        if args.spherical:
            # try to rotate the reynolds stress tensor
            pass

        if self.dataprefix == 'prime':
            avsolnf = args.solnf.replace(
                args.solnfbase,'{0}_n{1}_incr{2}_reynolds'
                                .format(args.solnfbase,args.nsolnf,args.incrsolnf)
            )
        else:
            avsolnf = args.solnf.replace(
                args.solnfbase,'{0}_n{1}_incr{2}'
                                .format(args.solnfbase,args.nsolnf,args.incrsolnf)
            )

        # save the pyfr format for the averaged files averaged files are not regional
        statstr0 = self.stats.get('data','fields')
        statstr = self.stats.get('data','fields')
        # update the solution fields
        if self.dataprefix == 'prime':
            if self.systemscls.name == 'navier-stokes':
                statstr = statstr +','+'prime-uu'+','+'prime-vv'+','+'prime-ww' +','+'prime-uv'+','+'prime-uw'+','+'prime-vw' 

        import h5py
        with h5py.File(avsolnf,'w') as f:
            # write the metadata
            for m in metadata:
                v = self.soln[m]
                if m == 'stats':
                    v=v.replace(statstr0,statstr)
                f[m] = np.array(v,dtype='S')

            # write the local data
            for part in partname:
                v = self.soln[part]
                if 'idxs' not in part:
                    v =solndata[part]
                f[part] = v
        # if in silent mode only output the pyfrs files
        import sys
        sys.exit()

    def write_fluctuations(self, args):
        # here only output the pyfr file of the fluctuations
        # do not output the vtu file
        if args.kte:
            import numpy as np
            items = list(self.soln)
            allpartname = [a for a in items if isinstance(self.soln[a],np.ndarray) ]

            partname   = (
                [a for a in items 
                    if isinstance(self.soln[a],np.ndarray) and 'idxs' not in a]
            )

            #partname = [a for a in self.soln_inf.keys()]

            avpartname = [a.replace('soln','tavg') for a in partname]

            metadata   = [a for a in items if isinstance(self.soln[a],str)]
                            
            #info   = self.soln['stats'].split('\n')
            #avinfo = self.avsoln['stats'].split('\n')
            info   = Inifile(self.soln['stats'])
            avinfo = Inifile(self.avsoln['stats'])

            #fields   = [item for item in info if 'fields' in item][0]
            #avfields = [item for item in avinfo if 'fields' in item][0]
            #avfields = avfields.replace('avg-','prime-')

            fields = info.get('data', 'fields')
            avfields = avinfo.get('data', 'fields')
            avfields = avfields.replace('avg-','prime-')

            import h5py
            with h5py.File('fluctuation_{}'.format(args.solnf),'w') as f:
                # write the metadata
                for m in metadata:
                    v = self.soln[m]
                    if m == 'stats':
                        v = v.replace('prefix = soln','prefix = prime')
                        v = v.replace(fields, avfields)
                    f[m] = np.array(v,dtype='S')

                # write the local data
                for part in allpartname:
                    v = self.soln[part]
                    if part in partname:
                        v  = v.swapaxes(0,1)
                        av = self.avsoln[part.replace('soln','tavg')]
                        # instantaneous file stores conserved variables
                        v = np.array(
                            self.elementscls.con_to_pri(v, self.cfg)
                        ).swapaxes(0,1)
                        
                        # partial store
                        if av.shape != v.shape:
                            ptpre, ptpost = part.rsplit('_',1)
                            av = av[:,:,self.soln[f'{ptpre}_idxs_{ptpost}']]
                        # do not substract density
                        # copy averaged rho into v
                        v[:, 0, :] = av[:, 0, :]
                        v[:, 1:, :] = v[:, 1:, :] - av[:, 1:, :]

                    f[part.replace('soln','prime')] = v
        import sys
        sys.exit()

    def tke_analysis(self):
        # do kte analysis
        # using smats etc to get the derivative 
        
        # for this calculation need both averaged and fluctuation field


        # <> for time average {} for density weighted average
        
        # Ma is small neglect compressibility for this analysis
        # see paper: Compressibility effects and turbulent kinetic
        # energy exchange in temporal mixing layers Ma>0.3 things starts to vary
        # σ_ij  density weighted averaged stress tensor
        # turbulent diffusion -1/2*∂{u''_j u''_i u''_i}/∂ x_j
        # production - {u''_i u''_j} ∂{u_i}/∂ x_j 
        # pressure diffusion -∂<p><u''_i>/ ∂ x_i
        # fluctuation pressure diffusion -∂<p'><u''_i>/ ∂ x_i
        # pressure dilation +  <p ∂ u''_i / ∂ x_i>
        # viscous transport + ∂<σ_ij u''_i>/ ∂ x_j
        # viscous dissipation -<σ_ij ∂ u''_i/ ∂ x_j> 
        
        for gid, ameshinfo in enumerate(self.mesh_inf):
            for partname, (etype, v) in ameshinfo.items():
                # construction an instance of the relevant elements class
                # element type name
                basiscls = subclass_where(BaseShape, name = etype)

                eles = self.elementscls(
                    basiscls, self.mesh[gid][partname], self.cfg, gid
                )

                smat = eles.smat_at_np('upts').transpose(2,0,1,3)
                rcpdjac = eles.rcpdjac_at_np('upts')
                gradop = eles.basis.m4.astype(self.dtype)
                
                # find the soln 

        pass
    
    def cal_turb_diffusion(self, smat, rcpdjac, gradop, soln):
        # calculate turbulent diffusion here
        # -1/2 ∂  u''_1 u''_i u''_i / ∂ x_1 
        # -1/2 ∂  u''_2 u''_i u''_i / ∂ x_2 
        # -1/2 ∂  u''_3 u''_i u''_i / ∂ x_3 

        # first evaluate (u''_1 u''_i u''_i)  (u''_2 u''_i u''_i) (u''_3 u''_i u''_i)
        


        # get the transformed gradient

        # get the untransformed gradient
        pass
    
    def cal_production(self):   
        # calculate production here
        # - u''_i u''_1 ∂ u_i / ∂ x_1
        # - u''_i u''_2 ∂ u_i / ∂ x_2
        # - u''_i u''_3 ∂ u_i / ∂ x_3

        # first evaluate (u''_1 u''_1) (u''_2 u''_1) (u''_3 u''_1) etc

        # get the transformed gradient

        # get the untransformed gradient
        pass

    def cal_pressure_diffusion(self):
        # - ∂ <p> <u''_i>/ ∂ x_i
        
        # get the transformed gradient

        # get the untransformed gradient
        pass
    def cal_fluctuation_pressiure_diffusion(self):
        # - ∂ <p'> <u''_i>/ ∂ x_i

        # get the transformed gradient

        # get the untransformed gradient
        pass
     
    def cal_pressure_dilation(self):
        # + <p> ∂ <u''_i> / ∂ x_i
        # get the transformed gradient

        # get the untransformed gradient
        pass
    
    def cal_viscous_transport(self):
        # calculate the stress from averaged field first 
        # viscous transport + ∂<σ_ij u''_i>/ ∂ x_j
        
        # return both stress and viscous transport
        pass

    def cal_viscous_dissipation(self):
        # viscous dissipation -<σ_ij ∂ u''_i/ ∂ x_j>
        # using stress to calculate transport

        #return viscous dissipation
        pass
