# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePlugin,RegionMixing
from pyfr.writers.native import NativeWriter


class WriterPlugin(BasePlugin, RegionMixing):
    name = 'writer'
    systems = ['*']
    formulations = ['dual', 'std']
    
    # writer will write the solutions on multiple meshes into one file
    # for visualization or post-processing, solutions will be reassigned 
    # into different files

    def __init__(self, intg, cfgsect, suffix=None):
        BasePlugin.__init__(self, intg, cfgsect, suffix)
        RegionMixing.__init__(self, intg, cfgsect)

        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')
        self._writer = NativeWriter(intg, self.mdata, basedir, basename)

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]
        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

        # If we're not restarting then write out the initial solution
        if not intg.isrestart:
            self.tout_last -= self.dt_out
            self(intg)

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return
        comm, rank, root = get_comm_rank_root()

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)
        # Prepare the metadata
        metadata = dict(intg.cfgmeta,
                        stats=stats.tostr(),
                        mesh_uuid=intg.mesh_uuid)
        # Prepare the data itself
        data = self.prepare_data(intg)
        # Write out the file
        solnfname = self._writer.write(data, metadata, intg.tcurr)

        mesh = intg.system.mesh
        zipmesh = [mesh] if isinstance(mesh,list) == False else mesh
        # If a post-action has been registered then invoke it
        # support multiple meshes
        self._invoke_postaction(mesh = [ a.fname for a in zipmesh], soln = solnfname,
                                t = intg.tcurr)
        # Update the last output time
        self.tout_last = intg.tcurr
