#ifndef HELPER_H
#define HELPER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "macros.hpp"
#include <iostream>

#define MAX_GRID_DIM 65535
#define N_EVENTS 6
#define N_STREAMS 17


static cudaStream_t stream_handles[N_STREAMS];
static cudaEvent_t  event_handles[N_EVENTS];

void initialize_stream_event();
void destroy_stream_event();

cudaStream_t get_stream_handle();
cudaEvent_t  get_event_handle();

void addrToCudaStream(unsigned long long int); 

void sync_device();

void tg_print_data(unsigned long long int datastart, 
                   unsigned long long int offset, 
                   unsigned int nums, int dtype); 


void get_nodal_basis_wrapper(int* cellIDs, double* rst, double* weights,
    double* xiGrid, int nFringe, int nSpts, int nSpts1D, int stream);

// for cells
void pack_cell_coords_wrapper(
    int* ucellIDs, int* ecellIDs, 
    double* xyz, double* coord_spts,
    unsigned int nCells, unsigned int nSpts, unsigned int nDims, 
    unsigned int soasz, unsigned int neled2, int stream = -1);

void unpack_unblank_u_wrapper(
    int* ucellIDs, int* ecellIDs,
    double* U_unblank, double* U_spts,
    unsigned int nCells,
    unsigned int nSpts, unsigned int nVars,
    unsigned int soasz, unsigned int neled2, int stream = -1);

// for fringe faces
void unpack_fringe_u_wrapper(
    double *U_fringe, double* U, 
    int* fringe_fpts, 
    unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, 
    unsigned int soasz, int stream = -1);

void unpack_fringe_grad_wrapper(
    double* dU_fringe, double* dU,
    unsigned int* fringe_fpts,
    unsigned int* dim_stride,
    unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims,
    unsigned int soasz, int stream = -1);

void pack_fringe_coords_wrapper(
    unsigned int* fringe_fpts, 
    double* xyz,
    double* coord_fpts, 
    int nPts, int nDims,
    unsigned int soasz,int stream = -1);

void reset_mpi_face_artbnd_status_wrapper(
    double* status, 
    int* mapping, 
    double val,
    unsigned int nface,
    unsigned int nfpts, unsigned int nvars, 
    unsigned int soasz, int strean=-1);

void move_grid_flat_wrapper(
    double* flatcoords, 
    double* flatcoords_ref, 
    unsigned int npts, 
    unsigned int ndims,
    double sgn,
    double* Rmat,
    double* offset,
    double* pivot,  // adding pivot to allow for arbrary pivot point
    int stream = -1); 

void move_grid_nested_wrapper(
    double* nestedcoords, 
    double* nestedcoords_ref,
    unsigned int ncells, // total number of elements
    unsigned int npts, // points per element
    unsigned int ndims,
    double sgn,
    double* Rmat,
    double* offset,
    double* pivot, // adding pivot info to allow for arbitrary pivot point
    int stream = -1);


void copy_to_mpi_rhs_wrapper(
    double* base, double* src,
    unsigned int* doffset, unsigned int* fidx,  // these two decide the offset for dest
    unsigned int* soffset, // this one decide the offset for src
    unsigned int* nfpts, unsigned int* fbase,
    unsigned int nvar, unsigned int nface, int stream = -1
);
 
/// functions for gmres etc
void pmg_helper(void (*pmgfunc) ());
#endif
