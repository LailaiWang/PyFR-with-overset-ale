#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
import itertools as it
import os

import mpi4py.rc
mpi4py.rc.initialize = False

import h5py

from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.progress_bar import ProgressBar
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_name, get_writer_by_extn


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', type=FileType('r'),
                           help='input mesh file')
    ap_import.add_argument('outmesh', help='output PyFR mesh file')
    types = sorted(cls.name for cls in subclasses(BaseReader))
    ap_import.add_argument('-t', dest='type', choices=types,
                           help='input file type; this is usually inferred '
                           'from the extension of inmesh')
    ap_import.set_defaults(process=process_import)

    # Partition command
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition.add_argument('np', help='number of partitions or a colon '
                              'delimited list of weights')
    ap_partition.add_argument('mesh', help='input mesh file')
    ap_partition.add_argument('solns', metavar='soln', nargs='*',
                              help='input solution files')
    ap_partition.add_argument('outd', help='output directory')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition.add_argument('-p', dest='partitioner', choices=partitioners,
                              help='partitioner to use')
    ap_partition.add_argument('-r', dest='rnumf', type=FileType('w'),
                              help='output renumbering file')
    ap_partition.add_argument('--popt', dest='popts', action='append',
                              default=[], metavar='key:value',
                              help='partitioner-specific option')
    ap_partition.add_argument('-t', dest='order', type=int, default=3,
                              help='target polynomial order; aids in '
                              'load-balancing mixed meshes')
    ap_partition.set_defaults(process=process_partition)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_export.add_argument('solnf', help='PyFR solution file to be converted')
    # to add more than one solution files
    # in case averaged files are saved with short time intervals
    ap_export.add_argument('-n', '--nsolnf', type = int, required = False, default = 1,
                           help='Number of solution files')
    ap_export.add_argument('-i', '--incrsolnf', type = int, required = False, default = 0,
                           help= 'Increment of solution files')
    # input the averaged data and instantaneous data to get the u'
    # if this is the case that an averaged file is supplied to check 
    # turbulent kinetci energy
    ap_export.add_argument('-blank','--blanking', action = 'store_true', 
                           help = 'whether to blank cell on back ground mesh')

    # to calculate u', add the instantaneous soln  file here
    ap_export.add_argument('-af', '--avsolnf',required = False, 
                           help='Averaged solution file')

    ap_export.add_argument('-sl', '--silent', required = False, type = bool,
                           help='Only output the pyfrs file')
    
    # note the difference between nis and n iis and i
    ap_export.add_argument('-nis','--ninstance', type = int, required=False, default = 1,
                           help = 'Number of time instances')
    ap_export.add_argument('-iis','--incrinst', type = int, required=False, default = 0,
                           help = 'Increment of time instances')
    
    # solnf idx base to infer the filename
    ap_export.add_argument('-fb','--solnfbase', type = str, required= False, 
                           help='solnf idx base for inferrring the files, eg: 030.00')

    ap_export.add_argument('outf', type=str, help='output file')

    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_export.add_argument('-t', dest='type', choices=types, required=False,
                           help='output file type; this is usually inferred '
                           'from the extension of outf')
    ap_export.add_argument('-d', '--divisor', type=int, default=0,
                           help='sets the level to which high order elements '
                           'are divided; output is linear between nodes, so '
                           'increased resolution may be required')
    ap_export.add_argument('-g', '--gradients', action='store_true',
                           help='compute gradients')
    ap_export.add_argument('-p', '--precision', choices=['single', 'double'],
                           default='single', help='output number precision; '
                           'defaults to single')
    # add support for multiple meshes
    ap_export.add_argument('-am','--addmsh', nargs = '+', required = False, default=None,
                  help = 'background mesh must be the first one, and additional meshes')
    ap_export.set_defaults(process=process_export)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    # main mesh
    ap_run.add_argument('mesh', help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    # load additional near body meshes
    ap_run.add_argument('-am','--addmsh', nargs = '+', required = False, default=None,
                         help = 'background mesh must be the first one, and additional meshes')
    ap_run.set_defaults(process=process_run)

    # Restart command
    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', help='mesh file')
    ap_restart.add_argument('soln', help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.add_argument('-am','--addmsh', nargs = '+', required = False, default=None,
                         help = 'background mesh must be the first one, and additional meshes')
    ap_restart.set_defaults(process=process_restart)

    # Options common to run and restart
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    for p in [ap_run, ap_restart]:
        p.add_argument('--backend', '-b', choices=backends, required=True,
                       help='backend to use')
        p.add_argument('--progress', '-p', action='store_true',
                       help='show a progress bar')

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


def process_import(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh)
    else:
        extn = os.path.splitext(args.inmesh.name)[1]
        reader = get_reader_by_extn(extn, args.inmesh)

    # Get the mesh in the PyFR format
    mesh = reader.to_pyfrm()

    # Save to disk
    with h5py.File(args.outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def process_partition(args):
    # Ensure outd is a directory
    if not os.path.isdir(args.outd):
        raise ValueError('Invalid output directory')

    # Partition weights
    if ':' in args.np:
        pwts = [int(w) for w in args.np.split(':')]
    else:
        pwts = [1]*int(args.np)

    # Partitioner-specific options
    opts = dict(s.split(':', 1) for s in args.popts)

    # Create the partitioner
    if args.partitioner:
        part = get_partitioner(args.partitioner, pwts, order=args.order,
                               opts=opts)
    else:
        for name in sorted(cls.name for cls in subclasses(BasePartitioner)):
            try:
                part = get_partitioner(name, pwts, order=args.order)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('No partitioners available')

    # Partition the mesh
    mesh, rnum, part_soln_fn = part.partition(NativeReader(args.mesh))

    # Prepare the solutions
    solnit = (part_soln_fn(NativeReader(s)) for s in args.solns)

    # Output paths/files
    paths = it.chain([args.mesh], args.solns)
    files = it.chain([mesh], solnit)

    # Iterate over the output mesh/solutions
    for path, data in zip(paths, files):
        # Compute the output path
        path = os.path.join(args.outd, os.path.basename(path.rstrip('/')))

        # Save to disk
        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f[k] = v

    # Write out the renumbering table
    if args.rnumf:
        print('etype,pold,iold,pnew,inew', file=args.rnumf)

        for etype, emap in sorted(rnum.items()):
            for k, v in sorted(emap.items()):
                print(','.join(map(str, (etype, *k, *v))), file=args.rnumf)


def process_export(args):
    # Get writer instance by specified type or outf extension
    # add a loop here 
    solnfs = [args.solnf]
    
    # this is to calculate fluctuations
    if args.ninstance > 1:
        inpyfrs = args.solnf
        solnfb  = args.solnfbase.split('.') 
        fidxs = [int(solnfb[0]) + args.incrinst*i for i in range(args.ninstance)]
        fidxstr = ['{:03}'.format(a) for a in fidxs]
        solnfs = [args.solnf.replace(solnfb[0], a) for a in fidxstr]

    for solnf in solnfs:
        args.solnf = solnf
        if args.type:
            writer = get_writer_by_name(args.type, args)
        else:
            extn = os.path.splitext(args.outf)[1]
            writer = get_writer_by_extn(extn, args)

        # Write the output file
        if args.silent is True: 
            print("Skip writing visualization file for soln {}".format(solnf)) 
        else:
            writer.write_out()


def _process_common(args, mesh, soln, cfg):
    
    # mesh could be instances or list of instances

    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()

    # Ensure MPI is suitably cleaned up
    register_finalize_handler()

    # Create a backend
    backend = get_backend(args.backend, cfg)

    # all meshes must be partitioned into same number of partitions ?
    # nope 
    rallocs = get_rank_allocation(mesh, cfg)

    # Construct the solver
    solver = get_solver(backend, rallocs, mesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and MPI.COMM_WORLD.rank == 0:
        pb = ProgressBar(solver.tstart, solver.tcurr, solver.tend)

        # Register a callback to update the bar after each step
        callb = lambda intg: pb.advance_to(intg.tcurr)
        solver.completed_step_handlers.append(callb)

    # Execute!
    solver.run()

    # Finalise MPI
    MPI.Finalize()


def process_run(args):
    if args.addmsh == None:
        _process_common(
            args, NativeReader(args.mesh), None, Inifile.load(args.cfg)
        )
    else:
        # read the primary mesh and additional mesh
        allmesh = [NativeReader(args.mesh)] + [NativeReader(a) for a in args.addmsh]

        _process_common(
            args, allmesh, None, Inifile.load(args.cfg)
        )


def process_restart(args):
    mesh = (
            [NativeReader(args.mesh)] + [NativeReader(a) for a in args.addmsh]
            if args.addmsh != None else
            NativeReader(args.mesh)
           )
    soln = NativeReader(args.soln)
    
    allmesh = [mesh] if isinstance(mesh, list) == False else mesh
    # Ensure the solution is from the mesh we are using
    if soln['mesh_uuid'].tolist() != [mesh['mesh_uuid'].encode() for mesh in allmesh]:
        raise RuntimeError('Invalid solution for mesh.')

    # Process the config file
    if args.cfg:
        cfg = Inifile.load(args.cfg)
    else:
        cfg = Inifile(soln['config'])

    _process_common(args, mesh, soln, cfg)


if __name__ == '__main__':
    main()
