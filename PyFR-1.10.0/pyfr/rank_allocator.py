# -*- coding: utf-8 -*-

from collections import defaultdict
import random
import re

from pyfr.mpiutil import get_comm_rank_root
from pyfr.util import subclass_where


def get_rank_allocation(mesh, cfg):
    name = cfg.get('backend', 'rank-allocator', 'linear')

    return subclass_where(BaseRankAllocator, name=name)(mesh, cfg)

# modification
# for multiple grids, each process also possesses only one partition

class BaseRankAllocator(object):
    name = None
    def __init__(self, mesh, cfg):
        self.cfg = cfg

        comm, rank, root = get_comm_rank_root()

        # Have the root rank determine the connectivity of the mesh
        if rank == root:
            if isinstance(mesh,list) == False:
                prankconn = self._get_mesh_connectivity(mesh)
                nparts = len(prankconn)
                granks = [nparts]
                grankoffsets = [0]
                grankrange = [nparts]
                if nparts != comm.size:
                    raise RuntimeError('Mesh has {0} partitions but running with '
                                       '{1} MPI ranks'.format(nparts, comm.size))

            else:
                # each process possess one partition
                gridranks = [0]
                prankconn = [ ]
                for idx, amesh in enumerate(mesh):
                    pconn = self._get_mesh_connectivity(amesh, gridranks[idx])
                    nprts = len(pconn)
                    gridranks.append(nprts)
                    prankconn = prankconn + pconn 
                # sanity check, 
                if sum(gridranks) != comm.size:
                    raise RuntimeError('Mesh has {0} paritions but running with '
                                       '{1} MPI ranks',format(sum(gridranks),comm.size))
                # remove the first element
                granks = gridranks[1:]
                grankrange = [sum(granks[:i+1]) for i in range(len(granks))]
                grankoffsets = [0] + grankrange[:-1]
        else:
            prankconn = None
            granks = None
            grankrange = None
            grankoffsets = None


        # Get subclass dependant info about each rank (e.g., hostname)
        rinfo = comm.gather(self._get_rank_info(), root=root)

        # If we are the root rank then perform the rank allocation
        if rank == root:
            mprankmap = self._get_mprankmap(prankconn, rinfo)
        else:
            mprankmap = None

        # Broadcast the connectivity and rank mappings to all other ranks
        self.prankconn = prankconn = comm.bcast(prankconn, root=root)
        self.mprankmap = mprankmap = comm.bcast(mprankmap, root=root)
        # each grid has m ranks 
        self.granks = granks = comm.bcast(granks, root=root)
        # rank range of each grid
        self.grankrange = grankrange = comm.bcast(grankrange, root=root)
        # offsets
        self.grankoffsets = grankoffsets = comm.bcast(grankoffsets, root=root)
        
        # Invert the mapping to obtain the physical-to-MPI rank mapping
        # This mapping is consistent among all meshes
        self.pmrankmap = sorted(range(comm.size), key=mprankmap.__getitem__)

        # Compute our physical rank
        self.prank = mprankmap[rank]

    def _get_mesh_connectivity(self, mesh, offset = 0):
        conn = defaultdict(list)
        for f in mesh:
            m = re.match(r'con_p(\d+)p(\d+)$', f)
            if m:
                lhs, rhs = int(m.group(1)), int(m.group(2))
                conn[lhs].append(rhs)

                if 'con_p{0}p{1}'.format(rhs, lhs) not in mesh:
                    raise ValueError('MPI interface ({0}, {1}) is not '
                                     'symmetric'.format(lhs, rhs))
        import numpy as np
        return [list(np.array(conn[i])+offset) for i in range(len(conn) or 1)]

    def _get_rank_info(self):
        pass

    def _get_mprankmap(self, prankconn, rinfo):
        pass


class LinearRankAllocator(BaseRankAllocator):
    name = 'linear'

    def _get_rank_info(self):
        return None

    def _get_mprankmap(self, prankconn, rinfo):
        return list(range(len(rinfo)))


class RandomRankAllocator(BaseRankAllocator):
    name = 'random'

    def _get_rank_info(self):
        return None

    def _get_mprankmap(self, prankconn, rinfo):
        return random.sample(range(len(rinfo)), len(rinfo))
