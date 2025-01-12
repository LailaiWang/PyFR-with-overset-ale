# -*- coding: utf-8 -*-

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where
from pyfr.writers.merge_overset import PostOverset

class BaseWriter(object):
    def __init__(self, args):
        from pyfr.solvers.base import BaseSystem

        # infer outf names names if addmsh for overset
        import os
        extn = os.path.splitext(args.outf)[1]
        outf = os.path.splitext(args.outf)[0]

        nmsh = 1 if args.addmsh == None else 1 + len(args.addmsh)
        self.outf = ['{0}-grid-{1}{2}'.format(outf,idx,extn) for idx in range(nmsh)]
        
        meshf = [args.meshf] if args.addmsh == None else [args.meshf]+args.addmsh

        # create a list of meshes 
        self.mesh = [NativeReader(a) for a in meshf] 

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
        
        if nmsh > 1 :
            if args.nsolnf == 1 :
                self.postoverset = PostOverset(
                    self.elementscls, self.mesh, self.soln, self.mesh_inf, self.soln_inf,
                    self.cfg, self.stats
                )
            else:
                base = int(args.solnf[-9:-6])
                filenames = [args.solnf[:-9] + f'{base + args.incrsolnf *i :03}' + '.pyfrs' for i in range(args.nsolnf)] 
                for file in filenames:
                    soln = NativeReader(file)
                    stats = Inifile(soln['stats'])
                    PostOverset(
                        self.elementscls, self.mesh, soln, self.mesh_inf, self.soln_inf,
                        self.cfg, stats
                    )
                import sys
                sys.exit('Existing batch post-processing')
                

