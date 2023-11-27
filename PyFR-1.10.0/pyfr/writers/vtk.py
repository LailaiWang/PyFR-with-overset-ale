# -*- coding: utf-8 -*-

from collections import defaultdict
import os
import re

import numpy as np
import math
from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where
from pyfr.writers import BaseWriter
from pyfr.solvers.moving_grid.movgrid import motion_exprs, calc_motion
from pyfr.quadrules import get_quadrule

class VTKWriter(BaseWriter):
    # Supported file types and extensions
    name = 'vtk'
    extn = ['.vtu', '.pvtu']
    
    # one solution file contains the solutions on one or more than one meshes
    # for visualization split the solution

    def __init__(self, args):
        super().__init__(args)
        self.vort=0
        self.mass=0
        self.mesh1=None
        self.dtype = np.dtype(args.precision).type
        self.divisor = args.divisor or self.cfg.getint('solver', 'order')

        # Solutions need a separate processing pipeline to other data
        if self.dataprefix == 'soln':
            self._pre_proc_fields = self._pre_proc_fields_soln
            self._post_proc_fields = self._post_proc_fields_soln
            self._soln_fields = list(self.elementscls.privarmap[self.ndims])
            self._vtk_vars = list(self.elementscls.visvarmap[self.ndims])
        # Otherwise we're dealing with simple scalar data
        else:
            self._pre_proc_fields = self._pre_proc_fields_scal
            self._post_proc_fields = self._post_proc_fields_scal
            self._soln_fields = self.stats.get('data', 'fields').split(',')
            self._vtk_vars = [(k, [k]) for k in self._soln_fields]

        self.blanking = args.blanking
        self.region = args.region
        self.calc_vorticity =args.vorticity
        if (self.region):
            self.blanking = False

        self.gid0pn=0
        self.pext={}

        # See if we are computing gradients
        if args.gradients:
            self._pre_proc_fields_ref = self._pre_proc_fields
            self._pre_proc_fields = self._pre_proc_fields_grad
            self._post_proc_fields = self._post_proc_fields_grad

            # Update list of solution fields
            self._soln_fields.extend(
                f'{f}-{d}'
                for f in list(self._soln_fields) for d in range(self.ndims)
            )

            # Update the list of VTK variables to solution fields
            nf = lambda f: [f'{f}-{d}' for d in range(self.ndims)]
            for var, fields in list(self._vtk_vars):
                if len(fields) == 1:
                    self._vtk_vars.append((f'grad {var}', nf(fields[0])))
                else:
                    self._vtk_vars.extend(
                        (f'grad {var} {f}', nf(f)) for f in fields
                    )
        # build up boundary surfaces
        self.bcparts = self._fetch_bc_info()

    def _pre_proc_fields_soln(self, name, mesh, soln):
        # Convert from conservative to primitive variables
        return np.array(self.elementscls.con_to_pri(soln, self.cfg))

    def _pre_proc_fields_scal(self, name, mesh, soln):
        return soln

    def _post_proc_fields_soln(self, vsoln):
        # Primitive and visualisation variable maps
        privarmap = self.elementscls.privarmap[self.ndims]
        visvarmap = self.elementscls.visvarmap[self.ndims]

        # Prepare the fields
        fields = []
        for fnames, vnames in visvarmap:
            ix = [privarmap.index(vn) for vn in vnames]

            fields.append(vsoln[ix])

        return fields

    def _post_proc_fields_scal(self, vsoln):
        return [vsoln[self._soln_fields.index(v)] for v, _ in self._vtk_vars]

    def _pre_proc_fields_grad(self, name, mesh, soln):
        # Call the reference pre-processor
        soln = self._pre_proc_fields_ref(name, mesh, soln)

        # Dimensions
        nvars, nupts = soln.shape[:2]

        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)

        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)

        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.swapaxes(0, 1).reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')
        gradsoln = gradsoln.reshape(nvars*self.ndims, nupts, -1)

        return np.vstack([soln, gradsoln])

    def calc_vort(self, name, mesh, soln, gid, mesh1=None):
        # Call the reference pre-processor
        #soln = self._pre_proc_fields_ref(name, mesh, soln)
        #def meanl:
        def _rcpdjac_at_np(eles,r):
            _, djacs_mpts = eles._smats_djacs_mpts

            # Interpolation matrix to pts
            m00 = eles.basis.mbasis.nodal_basis_at(r)

            # Interpolate the djacs
            djac = m00 @ djacs_mpts

            if np.any(djac < -1e-5):
                raise RuntimeError('Negative mesh Jacobians detected')

            return 1.0 / djac
        
        if gid==2:
            
            meanloc1=np.mean(mesh1,axis=0)
            meanloc2=np.mean(mesh,axis=0)
            
            nel,_=meanloc1.shape
            ellist=[]
            x=.7
            for i in range(nel):
                if (meanloc2[i,0]-x)*(meanloc2[i,0]+x)>0 or (meanloc2[i,1]-x)*(meanloc2[i,1]+x)>0 or (meanloc2[i,2]-x)*(meanloc2[i,2]+x)>0:
                    ellist.append(i)
            
            soln=soln[:,:,ellist]
            mesh=mesh[:,ellist,:]
            
        # Dimensions
        nvars, nupts = soln.shape[:2]
        
        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)
        
        
        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg, 0)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)
        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.swapaxes(0, 1).reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)
        ename='hex'
        qrule='gauss-legendre'
        qdeg = 6
        r = get_quadrule(ename, qrule, qdeg=qdeg)
        m0 = eles.basis.ubasis.nodal_basis_at(r.pts)
        
        #ploc=eles.ploc_at_np('upts').swapaxes(1,2)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')

        ggrad=gradsoln[1:4,:,:,:]
        nx,ny,nz,nw= ggrad.shape
        vort=np.zeros((nx,nz,nw))
        vort[0,:,:]=ggrad[1,2,:,:]-ggrad[2,1,:,:]
        vort[1,:,:]=ggrad[0,2,:,:]-ggrad[2,0,:,:]
        vort[2,:,:]=ggrad[1,0,:,:]-ggrad[0,1,:,:]
               
        vort1=np.square(vort)
        vort2=np.sum(vort1,axis=0)
        vort2=np.einsum('ij,...jk->...ik', m0, vort2)
        rr=soln[0,:,:]
        rr2=np.einsum('ij,...jk->...ik', m0, rr)

        
        rcpdjacs = _rcpdjac_at_np(eles,r.pts)
        rw=r.wts[:, None] / rcpdjacs
        
        iex=rw*vort2
        rr2=rw*rr2
        vort_sum=np.sum(iex)
        rho_sum=np.sum(rr2)
          
    

        return(rho_sum,vort_sum)

    def calc_error(self, name, mesh, soln,t):
        
        
        def my_func(a):
            theta=math.sin(np.pi*(a[0]+a[1]+a[2]))    

            return ((theta+3.0)*(theta+3.0))
            #return(1.0)
        
        def normalize(a):

            b=np.square(a)
            return(b)
            
        
        def _rcpdjac_at_np(eles,r):
            _, djacs_mpts = eles._smats_djacs_mpts

            # Interpolation matrix to pts
            m00 = eles.basis.mbasis.nodal_basis_at(r)

            # Interpolate the djacs
            djac = m00 @ djacs_mpts

            if np.any(djac < -1e-5):
                raise RuntimeError('Negative mesh Jacobians detected')

            return 1.0 / djac

        
        #soln=np.array(self.elementscls.con_to_pri(soln, self.cfg))
        nvars, nupts, neles = soln.shape
        ndims=mesh.shape[2]
        basiscls = subclass_where(BaseShape, name=name)
        eles = self.elementscls(basiscls, mesh, self.cfg, 0)
        ename='hex'
        qrule='gauss-legendre'
        qdeg = 6
        r = get_quadrule(ename, qrule, qdeg=qdeg)
        m0 = eles.basis.ubasis.nodal_basis_at(r.pts)
        
        ploc=eles.ploc_at_np('upts').swapaxes(1,2)
        ann_soln1=np.apply_along_axis(my_func, 2, ploc)
        error1=ann_soln1-soln[4,:,:]
        error2=normalize(error1)
        print(np.sum(error2)/(neles*nupts)*8.0)
        
        #update ploc and soln with quadrature   
        soln=np.einsum('ij,...jk->...ik', m0, soln)     
        ploc=np.einsum('ij,jk...->ik...', m0, ploc)
        
        # Get the ann soln
        ann_soln=np.apply_along_axis(my_func, 2, ploc)
        #error based on energy
        error=ann_soln-soln[4,:,:]
        

        err0=normalize(error)
        # calculate det of jacobians
        rcpdjacs = _rcpdjac_at_np(eles,r.pts)
        rw=r.wts[:, None] / rcpdjacs
        iex=rw*err0
        err_sum=np.sum(iex)
        print(err_sum)        
       



    def _post_proc_fields_grad(self, vsoln):
        # Prepare the fields
        fields = []
        for vname, vfields in self._vtk_vars:
            ix = [self._soln_fields.index(vf) for vf in vfields]

            fields.append(vsoln[ix])

        return fields

    def _get_npts_ncells_nnodes(self, sk):
        etype, neles = self.soln_inf[sk][0], self.soln_inf[sk][1][2]
        
        if 'overset' in self.cfg.sections():
            # check if I need to blank the blanked elements
            if self.blanking is True:
                skpre, skpost = sk.rsplit('_', 1)
                unblanked = self.soln[f'{skpre}_blank_{skpost}']
                idxunblanked = [idx for idx, stat in enumerate(unblanked) if stat == 1]

                neles = len(idxunblanked)

        etype = etype.split('-')[0]
        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)
        subdvcls = subclass_where(BaseShapeSubDiv, name=etype)

        # Number of vis points
        npts = shapecls.nspts_from_order(self.divisor + 1)*neles

        # Number of sub cells and nodes
        ncells = len(subdvcls.subcells(self.divisor))*neles
        nnodes = len(subdvcls.subnodes(self.divisor))*neles

        return npts, ncells, nnodes

    def _get_array_attrs(self, sk=None):
        dtype = 'Float32' if self.dtype == np.float32 else 'Float64'
        dsize = np.dtype(self.dtype).itemsize

        vvars = self._vtk_vars

        names = ['', 'connectivity', 'offsets', 'types','partition']
        types = [dtype, 'Int32', 'Int32', 'UInt8', 'Int32']
        comps = ['3', '', '', '', '1']

        for fname, varnames in vvars:
            names.append(fname.title())
            types.append(dtype)
            comps.append(str(len(varnames)))

        # If a solution has been given the compute the sizes
        if sk:
            npts, ncells, nnodes = self._get_npts_ncells_nnodes(sk)
            nb = npts*dsize

            sizes = [3*nb, 4*nnodes, 4*ncells, ncells,4*ncells]
            sizes.extend(len(varnames)*nb for fname, varnames in vvars)

            return names, types, comps, sizes
        else:
            return names, types, comps

    @memoize
    def _get_shape(self, name, nspts):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, self.cfg)

    @memoize
    def _get_std_ele(self, name, nspts):
        return self._get_shape(name, nspts).std_ele(self.divisor)

    @memoize
    def _get_mesh_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    @memoize
    def _get_soln_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)
    
    def write_out(self):
        from collections import OrderedDict

        ngparts = [0]
        # grid id is added
        for gid, amesh in enumerate(self.mesh):
            # extract the solution info from self.soln_inf
            
            # need to be compatible with previous output
            #soln_inf = OrderedDict((k, v) for k,v in self.soln_inf.items() 
            #                      if 'g{}'.format(gid) in k)
            soln_inf={}
            is_old = all(['-g' in k for k in self.soln_inf.keys()]) == False
            if (len(self.mesh)<2): 
                is_old=True
 
            if is_old:
                # this is for multi grids output
                soln_inf = OrderedDict((k, v) for k,v in self.soln_inf.items())
            else:
                # this is for multi grids output
                soln_inf = OrderedDict((k, v) for k,v in self.soln_inf.items() 
                                  if 'g{}'.format(gid) in k)
            if not soln_inf=={}: 
                ngparts.append(len(soln_inf))
                # build the offset 
                offset = sum(ngparts[:gid+1])
                # call function to write one mesh and corresponding soln
                self._write_out(
                    self.outf[gid], amesh, self.mesh_inf[gid], soln_inf, gid, offset
                )

    def _write_out(self, outf, mesh, mesh_inf, soln_inf, gid, goffset):
        name, extn = os.path.splitext(outf)
        parallel = extn == '.pvtu'

        lpart=[]

        parts = defaultdict(list)
        for sk, (etype, shape) in soln_inf.items():
            #part = sk.split('_')[-1]
            part = sk.split('_p')[-1]
                        
            lpart.append(int(part))

            pname = f'{name}_p{part}.vtu' if parallel else outf
            
            # mesh part name does not include g
            #partoff = 'p{}'.format(int(part[1:])-goffset)
            etypem = etype.split('-g{}'.format(gid))[0]
            parts[pname].append((part,f'spt_{etypem}_p{part}', sk))
        self.pext[gid]=min(lpart)


        if gid==0: self.gid0pn=int(part)+1     
        write_s_to_fh = lambda s: fh.write(s.encode())

        for pfn, misil in parts.items():
            with open(pfn, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="UnstructuredGrid" '
                              'version="0.1">\n<UnstructuredGrid>\n')

                # Running byte-offset for appended data
                off = 0

                # Header
                for pn, mk, sk in misil:
                    off = self._write_serial_header(fh, sk, off)

                write_s_to_fh('</UnstructuredGrid>\n'
                              '<AppendedData encoding="raw">\n_')

                # Data
                for pn, mk, sk in misil:
                    
                    self._write_data(
                        fh, mk, sk, pn, gid, goffset, mesh, mesh_inf, soln_inf
                    )

                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

        if parallel:
            with open(outf, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="PUnstructuredGrid" '
                              'version="0.1">\n<PUnstructuredGrid>\n')

                # Header
                self._write_parallel_header(fh)

                # Constitutent pieces
                for pfn in parts:
                    write_s_to_fh('<Piece Source="{0}"/>\n'
                                  .format(os.path.basename(pfn)))

                write_s_to_fh('</PUnstructuredGrid>\n</VTKFile>\n')
        # save the bc data


    def _write_darray(self, array, vtuf, dtype):
        array = array.astype(dtype)

        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _process_name(self, name):
        return re.sub(r'\W+', '_', name)

    def _write_serial_header(self, vtuf, sk, off):
        names, types, comps, sizes = self._get_array_attrs(sk)
        npts, ncells = self._get_npts_ncells_nnodes(sk)[:2]

        write_s = lambda s: vtuf.write(s.encode())
        #write_s(f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n')
        write_s(f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n'
                '<Points>\n')
        # Write vtk DaraArray headers
        '''
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s('<DataArray Name="{0}" type="{1}" '
                    'NumberOfComponents="{2}" '
                    'format="appended" offset="{3}"/>\n'
                    .format(self._process_name(n), t, c, off))
        '''
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s(f'<DataArray Name="{self._process_name(n)}" type="{t}" '
                    f'NumberOfComponents="{c}" '
                    f'format="appended" offset="{off}"/>\n')
       
            off += 4 + s

            # Write ends/starts of vtk file objects
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            elif i == 3:
                write_s('</Cells>\n<CellData>\n')
            elif i == 4:
                write_s('</CellData>\n<PointData>\n')

        # Write end of vtk element data
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_parallel_header(self, vtuf):
        names, types, comps = self._get_array_attrs()

        write_s = lambda s: vtuf.write(s.encode())
        write_s('<PPoints>\n')

        # Write vtk DaraArray headers
        for i, (n, t, s) in enumerate(zip(names, types, comps)):
            write_s('<PDataArray Name="{0}" type="{1}" '
                    'NumberOfComponents="{2}"/>\n'
                    .format(self._process_name(n), t, s))

            if i == 0:
                write_s('</PPoints>\n<PCells>\n')
            elif i == 3:
                write_s('</PCells>\n<PPointData>\n')

        write_s('</PPointData>\n')
    
    def _fetch_bc_info(self):
        # only output the wall boundary
        bcparts = [a for a in list(self.mesh) if 'bcon_' in a]
        return bcparts

    def _write_bc_data(self):
        pass

    def _write_data(self, vtuf, mk, sk, pn, gid, goffset, mesh, mesh_inf, soln_inf ):
        mk_g=mk
        
        # in case we have region
        if gid > 0 and self.region:
            pext=1
            pn_g=int(pn)-self.pext[gid]
            mk_g=mk.split('_p')[0]+"_p"+str(pn_g)
        elif (gid>0 and self.region==False ):
            pn_g=int(pn)-self.gid0pn
            if gid==2:
                pn_g=pn_g-1
            mk_g=mk.split('_p')[-1]
            mk_g=mk.split('_p')[0]+"_p"+str(pn_g)
        mesh0=mesh_inf
        name = mesh_inf[mk_g][0]
        mesh = mesh[mk_g].astype(self.dtype)
        soln = self.soln[sk].swapaxes(0, 1).astype(self.dtype)
        
        # Handle the case of partial solution files
        if soln.shape[2] != mesh.shape[1]:
            skpre, skpost = sk.rsplit('_', 1)

            mesh = mesh[:, self.soln[f'{skpre}_idxs_{skpost}'], :]


        # handle the case where overset blanking information is available
        # only output unblanked cells for visualization
        if not self.region:    
            if 'overset' in self.cfg.sections():
                skpre, skpost = sk.rsplit('_', 1)
                unblanked = self.soln[f'{skpre}_blank_{skpost}']
                idxunblanked = [idx for idx, stat in enumerate(unblanked) if stat == 1]
                # testing overset output
                if gid == 0 and self.blanking is False:
                    soln = self.postoverset.bksoln[sk].swapaxes(0,1)
                else:
                    mesh = mesh[:, idxunblanked, :]
                    soln = soln[:, :, idxunblanked]
            
        # Dimensions
        nspts, neles = mesh.shape[:2]
        
        if gid==1:
            self.mesh1=mesh


        #for tgv enstrophy
        ###########################
        if self.calc_vorticity:
            if self.blanking:
                if gid<2:
                    mass,vort=self.calc_vort(name, mesh, soln,gid)
                if gid==2:
                    mass,vort=self.calc_vort(name, mesh, soln, gid, self.mesh1)
                self.mass+=mass
                self.vort+=vort
                t = self.stats.get('solver-time-integrator','tcurr')
                
                if gid==1:
                    print(t,     self.vort/self.mass/2)

            else:
                if gid==0:
                    self.enstropy+=self.calc_vort(name, mesh, soln, gid)

            return
        #if gid==0:
        #    t = self.stats.get('solver-time-integrator','tcurr')
            #self.calc_error(name,mesh,soln,t)
            #exit()

        ############################

        # Sub divison points inside of a standard element
        svpts = self._get_std_ele(name, nspts)
        nsvpts = len(svpts)

        # Generate the operator matrices
        mesh_vtu_op = self._get_mesh_op(name, nspts, svpts)
        soln_vtu_op = self._get_soln_op(name, nspts, svpts)

        # Calculate node locations of VTU elements
        vpts = mesh_vtu_op @ mesh.reshape(nspts, -1)
        vpts = vpts.reshape(nsvpts, -1, self.ndims)
        
        if 'moving-object' in self.cfg.sections():
            motioninfo = motion_exprs(self.cfg, gid)
            if self.region:
                motioninfo = motion_exprs(self.cfg, gid)
            t = vpts.dtype.type(self.stats.get('solver-time-integrator','tcurr'))
            motion = calc_motion(t, t, motioninfo, vpts.dtype.type)
            rmat = np.array(motion['Rmat']).reshape(-1,3).astype(vpts.dtype)
            offset = np.array(motion['offset'])
            offset = np.tile(offset,(vpts.shape[0], vpts.shape[1],1))
            
            # moving the pivot point for rotation
            pivot = np.array(motion['pivot'])
            pivot = np.tile(pivot,(vpts.shape[0], vpts.shape[1],1))
            pivot = pivot + offset

            vpts = vpts + offset
            vpts = vpts.swapaxes(0,2)
            pivot = pivot.swapaxes(0,2)
            vpts = np.einsum('ij,jk...->ik...', rmat, vpts-pivot) + pivot
            vpts = vpts.swapaxes(0,2)


        # Pre-process the solution
        soln = self._pre_proc_fields(name, mesh, soln).swapaxes(0, 1)
        
        # Interpolate the solution to the vis points
        vsoln = soln_vtu_op @ soln.reshape(len(soln), -1)
        vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)
        
        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Write element node locations to file
        self._write_darray(vpts.swapaxes(0, 1), vtuf, self.dtype)

        # Perform the sub division
        subdvcls = subclass_where(BaseShapeSubDiv, name=name)
        nodes = subdvcls.subnodes(self.divisor)

        # Prepare VTU cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(subdvcls.subcelloffs(self.divisor), (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]

        # Tile VTU cell type numbers
        vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)

        # VTU cell partition numbers
        vtu_part = np.full_like(vtu_typ, pn)
        
        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, vtuf, np.int32)
        self._write_darray(vtu_off, vtuf, np.int32)
        self._write_darray(vtu_typ, vtuf, np.uint8)

        self._write_darray(vtu_part, vtuf, np.int32)
        # Process and write out the various fields
        for arr in self._post_proc_fields(vsoln):
            self._write_darray(arr.T, vtuf, self.dtype)


class BaseShapeSubDiv(object):
    vtk_types = dict(tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    vtk_nodes = dict(tri=3, quad=4, tet=4, pyr=5, pri=6, hex=8)

    @classmethod
    def subcells(cls, n):
        pass

    @classmethod
    def subcelloffs(cls, n):
        return np.cumsum([cls.vtk_nodes[t] for t in cls.subcells(n)])

    @classmethod
    def subcelltypes(cls, n):
        return np.array([cls.vtk_types[t] for t in cls.subcells(n)])

    @classmethod
    def subnodes(cls, n):
        pass


class TensorProdShapeSubDiv(BaseShapeSubDiv):
    @classmethod
    def subnodes(cls, n):
        conbase = np.array([0, 1, n + 2, n + 1])

        # Extend quad mapping to hex mapping
        if cls.ndim == 3:
            conbase = np.hstack((conbase, conbase + (1 + n)**2))

        # Calculate offset of each subdivided element's nodes
        nodeoff = np.zeros((n,)*cls.ndim, dtype=np.int32)
        for dim, off in enumerate(np.ix_(*(range(n),)*cls.ndim)):
            nodeoff += off*(n + 1)**dim

        # Tile standard element node ordering mapping, then apply offsets
        internal_con = np.tile(conbase, (n**cls.ndim, 1))
        internal_con += nodeoff.T.flatten()[:, None]

        return np.hstack(internal_con)


class QuadShapeSubDiv(TensorProdShapeSubDiv):
    name = 'quad'
    ndim = 2

    @classmethod
    def subcells(cls, n):
        return ['quad']*(n**2)


class HexShapeSubDiv(TensorProdShapeSubDiv):
    name = 'hex'
    ndim = 3

    @classmethod
    def subcells(cls, n):
        return ['hex']*(n**3)


class TriShapeSubDiv(BaseShapeSubDiv):
    name = 'tri'

    @classmethod
    def subcells(cls, n):
        return ['tri']*(n**2)

    @classmethod
    def subnodes(cls, n):
        conlst = []

        for row in range(n, 0, -1):
            # Lower and upper indices
            l = (n - row)*(n + row + 3) // 2
            u = l + row + 1

            # Base offsets
            off = [l, l + 1, u, u + 1, l + 1, u]

            # Generate current row
            subin = np.ravel(np.arange(row - 1)[..., None] + off)
            subex = [ix + row - 1 for ix in off[:3]]

            # Extent list
            conlst.extend([subin, subex])

        return np.hstack(conlst)


class TetShapeSubDiv(BaseShapeSubDiv):
    name = 'tet'

    @classmethod
    def subcells(cls, nsubdiv):
        return ['tet']*(nsubdiv**3)

    @classmethod
    def subnodes(cls, nsubdiv):
        conlst = []
        jump = 0

        for n in range(nsubdiv, 0, -1):
            for row in range(n, 0, -1):
                # Lower and upper indices
                l = (n - row)*(n + row + 3) // 2 + jump
                u = l + row + 1

                # Lower and upper for one row up
                ln = (n + 1)*(n + 2) // 2 + l - n + row
                un = ln + row

                rowm1 = np.arange(row - 1)[..., None]

                # Base offsets
                offs = [(l, l + 1, u, ln), (l + 1, u, ln, ln + 1),
                        (u, u + 1, ln + 1, un), (u, ln, ln + 1, un),
                        (l + 1, u, u+1, ln + 1), (u + 1, ln + 1, un, un + 1)]

                # Current row
                conlst.extend(rowm1 + off for off in offs[:-1])
                conlst.append(rowm1[:-1] + offs[-1])
                conlst.append([ix + row - 1 for ix in offs[0]])

            jump += (n + 1)*(n + 2) // 2

        return np.hstack([np.ravel(c) for c in conlst])


class PriShapeSubDiv(BaseShapeSubDiv):
    name = 'pri'

    @classmethod
    def subcells(cls, n):
        return ['pri']*(n**3)

    @classmethod
    def subnodes(cls, n):
        # Triangle connectivity
        tcon = TriShapeSubDiv.subnodes(n).reshape(-1, 3)

        # Layer these rows of triangles to define prisms
        loff = (n + 1)*(n + 2) // 2
        lcon = [[tcon + i*loff, tcon + (i + 1)*loff] for i in range(n)]

        return np.hstack([np.hstack(l).flat for l in lcon])


class PyrShapeSubDiv(BaseShapeSubDiv):
    name = 'pyr'

    @classmethod
    def subcells(cls, n):
        cells = []

        for i in range(n, 0, -1):
            cells += ['pyr']*(i**2 + (i - 1)**2)
            cells += ['tet']*(2*i*(i - 1))

        return cells

    @classmethod
    def subnodes(cls, nsubdiv):
        lcon = []

        # Quad connectivity
        qcon = [QuadShapeSubDiv.subnodes(n + 1).reshape(-1, 4)
                for n in range(nsubdiv)]

        # Simple functions
        def _row_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*i + j + 1)
                             for i in range(a, n + b)
                             for j in range(n - 1)])

        def _col_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*(i + 1) + j)
                             for i in range(n - 1)
                             for j in range(a, n + b)])

        u = 0
        for n in range(nsubdiv, 0, -1):
            l = u
            u += (n + 1)**2

            lower_quad = qcon[n - 1] + l
            upper_pts = np.arange(n**2) + u

            # First set of pyramids
            lcon.append([lower_quad, upper_pts])

            if n > 1:
                upper_quad = qcon[n - 2] + u
                lower_pts = np.hstack([range(k*(n + 1)+1, (k + 1)*n + k)
                                       for k in range(1, n)]) + l

                # Second set of pyramids
                lcon.append([upper_quad[:, ::-1], lower_pts])

                lower_row = _row_in_quad(n + 1, 1, -1) + l
                lower_col = _col_in_quad(n + 1, 1, -1) + l

                upper_row = _row_in_quad(n) + u
                upper_col = _col_in_quad(n) + u

                # Tetrahedra
                lcon.append([lower_col, upper_row])
                lcon.append([lower_row[:, ::-1], upper_col])

        return np.hstack([np.column_stack(l).flat for l in lcon])
