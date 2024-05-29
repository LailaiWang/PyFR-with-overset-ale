#include "helper.h"

void initialize_stream_event() {
  stream_handles[0] = cudaStreamPerThread;
  for(int i=1;i<N_STREAMS-1;i++) {
    cudaStreamCreate(&stream_handles[i]);
  }

  for(int i=0;i<N_EVENTS;i++) {
    cudaEventCreateWithFlags(&event_handles[i], cudaEventDisableTiming);
  }

}

void destroy_stream_event() {

}

cudaStream_t get_stream_handle() {
    return stream_handles[3];
}

cudaEvent_t get_event_handle() {
    return event_handles[0];
}

/* Convert the pycuda stream handle to c++ syntax
 * and store it in pre-existing storage
 */
void addrToCudaStream(unsigned long long int addr) {
    stream_handles[N_STREAMS] = reinterpret_cast<cudaStream_t> (addr);
}

void sync_device() {
    cudaDeviceSynchronize();
}

__global__
void print_data_kernel_int(int* data, int n) {
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < n) 
        printf("thread %d data is %d\n",idx,data[idx]);

}

// for float
__global__
void print_data_kernel_float(float* data, int n) {
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < n) 
        printf("thread %d data is %f\n",idx,data[idx]);

}

// for double
__global__
void print_data_kernel_double(double* data, int n) {
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < n) 
        printf("thread %d data is %15.7e\n",idx,data[idx]);

}

void tg_print_data(unsigned long long int datastart, 
                   unsigned long long int offset, 
                   unsigned int nums, int dtype) {
    // given offset 
    // given start
    
    unsigned long long int data = datastart + offset;
    if(dtype == 0) {
        int* d = reinterpret_cast<int*> (data);
        print_data_kernel_int<<<ceil(nums/128.0), 128>>> (d, nums);
    } else if (dtype == 1) {
        float* d = reinterpret_cast<float*> (data);
        print_data_kernel_float<<<ceil(nums/128.0), 128>>> (d, nums);
    } else {
        double* d = reinterpret_cast<double*> (data);
        print_data_kernel_double<<<ceil(nums/128.0), 128>>> (d, nums);
    }

}

__device__ __forceinline__
double Lagrange_gpu(double* xiGrid, int npts, double xi, int mode)
{
  double val = 1.0;

  for (int i = 0; i < mode; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  for (int i = mode + 1; i < npts; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  return val;
}

template<int nSpts1D>
__global__
void get_nodal_basis(double* rst_in, double* weights,
    double* xiGrid, int nFringe)
{
  const int nSpts = nSpts1D*nSpts1D*nSpts1D;
  const int idx = (blockDim.x * blockIdx.x + threadIdx.x);

  if (nSpts1D == 1)
  {
    for (int i = idx; i < nFringe * nSpts; i += gridDim.x * blockDim.x)
      weights[i] = 1.0;
    return;
  }

  __shared__ double xi[nSpts1D];

  if (threadIdx.x < nSpts1D)
    xi[threadIdx.x] = xiGrid[threadIdx.x];

  __syncthreads();

  for (int i = idx; i < nFringe * nSpts; i += gridDim.x * blockDim.x)
  {
    int spt = i % nSpts;
    int ipt = i / nSpts;

    if (ipt >= nFringe) continue;

    double rst[3];
    for (int d = 0; d < 3; d++)
      rst[d] = rst_in[3*ipt+d];

    int ispt = spt % nSpts1D;
    int jspt = (spt / nSpts1D) % nSpts1D;
    int kspt = spt / (nSpts1D*nSpts1D);
    weights[nSpts*ipt+spt] = Lagrange_gpu(xi,nSpts1D,rst[0],ispt) *
        Lagrange_gpu(xi,nSpts1D,rst[1],jspt) *
        Lagrange_gpu(xi,nSpts1D,rst[2],kspt);
  }
}



void get_nodal_basis_wrapper(int* cellIDs, double* rst, double* weights,
    double* xiGrid, int nFringe, int nSpts, int nSpts1D, int stream) {

  int threads = 128;
  int blocks = min((nFringe * nSpts + threads - 1) / threads, MAX_GRID_DIM);
  int nbShare = nSpts1D*sizeof(double);

  if (stream == -1) {
    switch (nSpts1D) {
      case 1:
        get_nodal_basis<1><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 2:
        get_nodal_basis<2><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 3:
        get_nodal_basis<3><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 4:
        get_nodal_basis<4><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 5:
        get_nodal_basis<5><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 6:
        get_nodal_basis<6><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      default:
        ThrowException("nSpts1D case not implemented");
    }
  } else {
    switch (nSpts1D) {
      case 1:
        get_nodal_basis<1><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 2:
        get_nodal_basis<2><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 3:
        get_nodal_basis<3><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 4:
        get_nodal_basis<4><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 5:
        get_nodal_basis<5><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 6:
        get_nodal_basis<6><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      default:
        ThrowException("nSpts1D case not implemented");
    }
  }

  check_error();
}

//template<int nDims>
// nCells and nSpts are values for same type of unblanked 
__global__
void pack_cell_coords(
    int* ucellIDs, int* ecellIDs,
    double* xyz, double* coord_spts,
    unsigned int nCells, unsigned int nSpts,
    unsigned int nDims, unsigned int soasz, unsigned int neled2) {
  
  // involve different element types such that the solution points each 
  // element type has are different

  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  //const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  //const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;
  const unsigned int spt = idx % nSpts;
  const unsigned int ele = idx / nSpts;
  if (ele >= nCells)
    return;

  int icu = ucellIDs[ele]; // stores the starting idx of the element_entry
  int ice = ecellIDs[ele];
  int isz = ice % soasz;
  int n2  = ice / soasz;
    
  // xyz is a flattened array (neles*Max_nspts*dim)
  // coords_spts has the shape of [nspts, neled2, dim, soasz]
  // index for xyz is eles*nspts*d + spt*d + d
  // index for coords_spts 
  // spt*neled2*dim*sosaz+n2*dim*soasz+d*soasz+n_soasz
  for (unsigned int d = 0; d < nDims; d++) {
    xyz[icu*nDims+spt*nDims+d] = coord_spts[spt*neled2*nDims*soasz+n2*nDims*soasz+d*soasz+isz];
  }
}

void pack_cell_coords_wrapper(
    int* ucellIDs, int* ecellIDs, 
    double* xyz, double* coord_spts,
    unsigned int nCells,
    unsigned int nSpts, unsigned int nDims,
    unsigned int soasz, unsigned int neled2, int stream) {

  int threads = 128;
  int blocks = (nCells * nSpts + threads - 1) / threads;

  if (stream == -1)
  {
    //if (nDims == 2)
    //  pack_cell_coords<2><<<blocks, threads>>>(cellIDs,xyz,coord_spts,nCells,nSpts);
    //else
    //  pack_cell_coords<3><<<blocks, threads>>>(cellIDs,xyz,coord_spts,nCells,nSpts);
    pack_cell_coords<<<blocks, threads>>>(ucellIDs, ecellIDs, xyz, 
                                          coord_spts, nCells, nSpts, nDims,
                                          soasz, neled2);
  }
  else
  {
    //if (nDims == 2)
    //  pack_cell_coords<2><<<blocks, threads, 0, stream_handles[stream]>>>(cellIDs,
    //      xyz,coord_spts,nCells,nSpts);
    //else
    //  pack_cell_coords<3><<<blocks, threads, 0, stream_handles[stream]>>>(cellIDs,
    //      xyz,coord_spts,nCells,nSpts);
    pack_cell_coords<<<blocks, threads, 0, stream_handles[stream]>>>(ucellIDs, ecellIDs,
        xyz, coord_spts, nCells, nSpts, nDims, soasz, neled2);
  }

  check_error();
}



__global__
void unpack_unblank_u(
    int* ucellIDs, int* ecellIDs, 
    double* U_unblank, double* U_spts, 
    unsigned int nCells, 
    unsigned int nSpts, unsigned int nVars,
    unsigned int soasz, unsigned int neled2)
{   
  // note the differences from pack_coords
  const unsigned int tot_ind = (blockDim.x * blockIdx.x + threadIdx.x);
  //const unsigned int var = tot_ind % nVars;
  //const unsigned int spt = (tot_ind / nVars) % nSpts;
  //const unsigned int ele = tot_ind / (nSpts * nVars);

  const unsigned int ele = tot_ind / (nSpts);
  const unsigned int spt = tot_ind % nSpts;
  if (ele >= nCells || spt >= nSpts)
    return;
  
  // here icu is the starting idx of cell spts
  int icu = ucellIDs[ele];
  int ice = ecellIDs[ele];
  int isz = ice % soasz;
  int n2  = ice / soasz;
  // datashape of u is [nspts, neled2, nvars, soasz]

  //U_spts(spt, var, ele) = U_unblank(ic,spt,var);
  for(unsigned int var =0;var<nVars;var++) {
      U_spts[spt*neled2*nVars*soasz+n2*nVars*soasz+var*soasz+isz] = U_unblank[icu*nVars+spt*nVars+var];
  }
  //printf("var is %d\n",var);
  //if(var == 1) {
  //  printf("v is %lf\e", U_unblank[icu*nVars+spt*nVars+var]);
  //}
}

void unpack_unblank_u_wrapper(
    int* ucellIDs, int* ecellIDs,
    double* U_unblank, double* U_spts,
    unsigned int nCells,
    unsigned int nSpts, unsigned int nVars,
    unsigned int soasz, unsigned int neled2, int stream) {

  int threads = 128;
  //int blocks = (nCells * nSpts * nVars + threads - 1) / threads;
  int blocks = (nCells * nSpts  + threads - 1) / threads;

  if (stream == -1) {
    unpack_unblank_u<<<blocks, threads>>>(ucellIDs, ecellIDs, 
            U_unblank, U_spts, nCells, nSpts, nVars, soasz, neled2);
  } else {
    unpack_unblank_u<<<blocks, threads, 0, stream_handles[stream]>>>(
        ucellIDs, ecellIDs, U_unblank, U_spts,
        nCells, nSpts, nVars, soasz, neled2);
  }

  check_error();
}


//template<int nDims>
__global__
void pack_fringe_coords(
    unsigned int* fringe_fpts, 
    double* xyz, double* coord_fpts, 
    int nPts, unsigned int soasz, int nDims) {

  const unsigned int pt = blockDim.x * blockIdx.x + threadIdx.x;
    
  if (pt >= nPts)
    return;

  unsigned int gfpt = fringe_fpts[pt];
  
  // data in PyFR [nfpts,neled2,dim,soasz]

  for (unsigned int d = 0; d < nDims; d++) {
    xyz[pt*nDims+d] = coord_fpts[gfpt+d*soasz];
  }
}

void pack_fringe_coords_wrapper(
    unsigned int* fringe_fpts,
    double* xyz, double* coord_fpts, 
    int nPts, int nDims, unsigned int soasz, int stream) {

  int threads = 128;
  int blocks = (nPts + threads - 1) / threads;

  if (stream == -1) {
    //if (nDims == 2)
    //  pack_fringe_coords<2><<<blocks, threads>>>(fringe_fpts,xyz,coord_fpts,nPts,soasz);
    //else {
      pack_fringe_coords<<<blocks, threads>>>(fringe_fpts,xyz,
        coord_fpts,nPts,soasz,nDims);
    //}
  } else {
    //if (nDims == 2)
    //  pack_fringe_coords<2><<<blocks, threads, 0, stream_handles[stream]>>>(fringe_fpts,
    //      xyz,coord_fpts,nPts,soasz);
    //else
      pack_fringe_coords<<<blocks, threads, 0, stream_handles[stream]>>>(fringe_fpts,
          xyz,coord_fpts,nPts,soasz,nDims);
  }

  check_error();
}

//template <unsigned nDims>
__global__
void unpack_fringe_grad(
    double* dU_fringe, double* dU, 
    unsigned int* fringe_fpts,
    unsigned int* dim_stride,
    unsigned int nFringe,
    unsigned int nFpts,
    unsigned int nVars, unsigned int nDims,
    unsigned int soasz) {
  
  // do not use face since fringe faces could be of different types
  //const unsigned int tot_ind = (blockDim.x * blockIdx.x + threadIdx.x);
  //const unsigned int var = tot_ind % nVars;
  //const unsigned int pt = tot_ind / nVars;
  //const unsigned int fpt = (tot_ind / nVars) % nFpts;
  //const unsigned int face = tot_ind / (nFpts * nVars);
  //if (fpt >= nFpts || face >= nFringe || var >= nVars || pt >= nFpts)
  //  return;
  
  const unsigned int tot_ind = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int var = tot_ind % nVars;
  const unsigned int pt = tot_ind / nVars;

  if (pt>=nFpts)
    return;

  unsigned int gft  = fringe_fpts[pt];
  unsigned int ds = dim_stride[pt];
  
  //datashape of dU is [ndims, nfpts, neled2, nvars, soasz]

  //const unsigned int gfpt = fringe_fpts(fpt, face);
  //const unsigned int side = fringe_side(fpt, face);

  for (unsigned int dim = 0; dim < nDims; dim++) {
    //dU(side, dim, var, gfpt) = dU_fringe(face, fpt, dim, var);
     dU[dim*ds+gft+var*soasz] =   dU_fringe[pt*nDims*nVars+dim*nVars+var];
  }
}

void unpack_fringe_grad_wrapper(
    double* dU_fringe, double* dU,
    unsigned int* fringe_fpts,
    unsigned int* dim_stride,
    unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims,
    unsigned int soasz, int stream) {

  int threads  = 128;
  //int blocks = (nFringe * nFpts * nVars + threads - 1) / threads;
  // here nFpts is the total number of fpts
  int blocks = (nFpts*nVars + threads-1)/threads;

  if (stream == -1)
  {
    //if (nDims == 2)
    //  unpack_fringe_grad<2><<<blocks, threads>>>(dU_fringe, dU, fringe_fpts,
    //                                     nFringe, nFpts, nVars, soasz);

    //else if (nDims == 3)
    //  unpack_fringe_grad<3><<<blocks, threads>>>(dU_fringe, dU, fringe_fpts,
    //                                     nFringe, nFpts, nVars, soasz);
      unpack_fringe_grad<<<blocks, threads>>>(dU_fringe, dU, fringe_fpts, dim_stride,
                                 nFringe, nFpts, nVars, nDims, soasz);
  }
  else
  {
    //if (nDims == 2)
    //  unpack_fringe_grad<2><<<blocks, threads, 0, stream_handles[stream]>>>
    //      (dU_fringe, dU, fringe_fpts, nFringe, nFpts, nVars, soasz);

    //else if (nDims == 3)
    //  unpack_fringe_grad<3><<<blocks, threads, 0, stream_handles[stream]>>>
    //      (dU_fringe, dU, fringe_fpts, nFringe, nFpts, nVars, soasz);
      unpack_fringe_grad<<<blocks, threads, 0, stream_handles[stream]>>>
          (dU_fringe, dU, fringe_fpts, dim_stride, nFringe, nFpts, nVars, nDims, soasz);
  }

  check_error();
}

__global__
void unpack_fringe_u(
    double* U_fringe, double* U,
    unsigned int* fringe_fpts,
    unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, unsigned int soasz)
{
  //const unsigned int tot_ind = (blockDim.x * blockIdx.x + threadIdx.x);
  //const unsigned int var = tot_ind % nVars;
  //const unsigned int fpt = (tot_ind / nVars) % nFpts;
  //const unsigned int pt = tot_ind / nVars;
  //const unsigned int face = tot_ind / (nFpts * nVars);
  const unsigned int pt = (blockDim.x * blockIdx.x + threadIdx.x);

  //if (face >= nFringe || pt >= nFpts || fpt >= nFpts)
  //  return;
  if(pt>=nFpts)
    return;
  
  //printf("thread id is %d pt is %d gft is %d\n", tot_ind, pt, fringe_fpts[pt] );

  unsigned int gft  = fringe_fpts[pt];

  // datashape of U is [nfpts, neled2, nvars, soasz] 
  //if (var == 1) printf("U_fringe is %lf\n",U_fringe[pt*nVars+var]);
  //U[gft+var*soasz] = U_fringe[pt*nVars+var];  
  for(unsigned int var = 0; var< nVars; var++) {
      U[gft+var*soasz] = U_fringe[pt*nVars+var];  
  }
}

void unpack_fringe_u_wrapper(
    double* U_fringe, double* U,
    unsigned int* fringe_fpts,
    unsigned int nFringe, 
    unsigned int nFpts, unsigned int nVars,
    unsigned int soasz, int stream)
{
  int threads = 128;
  //int blocks = (nFringe * nFpts * nVars + threads - 1) / threads;
  int blocks = ( nFpts  + threads - 1) / threads;

  if (stream == -1) {
    unpack_fringe_u<<<blocks, threads>>>(U_fringe, U, fringe_fpts,
        nFringe, nFpts, nVars, soasz);
  } else {
    unpack_fringe_u<<<blocks, threads, 0, stream_handles[stream]>>>(U_fringe, U,
        fringe_fpts, nFringe, nFpts, nVars, soasz);
  }

  check_error();
}

__global__
void reset_mpi_face_artbnd_status(
    double* status,
    unsigned int* mapping, 
    double val,
    unsigned int nface,
    unsigned int nfpts,
    unsigned int nvars, unsigned int soasz) {

    const unsigned int pt = (blockDim.x * blockIdx.x + threadIdx.x);
    if(pt >= nfpts) return;
    
    unsigned int gft = mapping[pt];

    for(unsigned int var=0; var < nvars; var++){
        status[gft+var*soasz] = val;
    }
}

void reset_mpi_face_artbnd_status_wrapper(
    double* status, 
    unsigned int* mapping,
    double val, 
    unsigned int nface,
    unsigned int nfpts, unsigned int nvars, unsigned int soasz, int stream) {
    
    const int threads = 256;
    int blocks = (nfpts + threads -1) / threads;
    if (stream == -1) {
        reset_mpi_face_artbnd_status<<<blocks, threads>>> (
            status, mapping, val, nface, nfpts, nvars, soasz
        );
    } else {
        reset_mpi_face_artbnd_status<<<blocks, threads, 0, stream_handles[stream]>>> (
            status, mapping, val, nface, nfpts, nvars, soasz
        );
    }   

}

__global__
void move_flat(
    double* flatcoords,
    double* flatcoords_ref,
    unsigned int npts,
    unsigned int ndims,
    double sgn,
    double* Rmat,
    double* offset,
    double* pivot) {
  const unsigned int pt = (blockDim.x * blockIdx.x + threadIdx.x);
  if(pt>= npts)
    return;
    
  double pvt[3] = {pivot[0], pivot[1], pivot[2]};

  //for(unsigned int d1 = 0; d1 <ndims; d1++) {
  //  pvt[d1] += offset[d1];
  //}

  double xtmp[3] = {0.0}; // fix it
  for(unsigned int d1 = 0; d1 < ndims; d1++) {
    xtmp[d1] = offset[d1];
    for(unsigned int d2 = 0; d2 < ndims;d2++) {
        //xtmp[d1] += Rmat[d1*ndims+d2]*(flatcoords_ref[pt*ndims+d2]+offset[d2]);
        xtmp[d1] += Rmat[d1*ndims+d2]*(flatcoords_ref[pt*ndims+d2]-pvt[d2]);
    }
  }

  for(unsigned int d = 0; d<ndims; d++)
    flatcoords[pt*ndims+d] = xtmp[d] + pvt[d];
}

void move_grid_flat_wrapper(
    double* flatcoords, 
    double* flatcoords_ref, 
    unsigned int npts, 
    unsigned int ndims,
    double sgn,
    double* Rmat,
    double* offset,
    double* pivot,
    int stream) {

  int threads = 128;
  int blocks = (npts + threads -1)/threads;

  if(stream == -1) {
    move_flat<<<blocks,threads>>>(
        flatcoords, flatcoords_ref, npts, ndims, sgn,Rmat,offset, pivot
    );
  } else {
    move_flat<<<blocks,threads,0,stream_handles[stream]>>>(
        flatcoords, flatcoords_ref, npts, ndims, sgn, Rmat, offset, pivot
    );
  }
  check_error();
}

__global__
void move_nested(
    double* nestedcoords, 
    double* nestedcoords_ref, 
    unsigned int ncells, 
    unsigned int npts,
    unsigned int ndims,
    double sgn, 
    double* Rmat,
    double* offset,
    double* pivot) {

  const unsigned int pt = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int nc = pt / npts;
  const unsigned int upt = pt % npts;

  if(nc >= ncells)
    return;
  // for nested ecoords, coord[ele+ncells*(d+ndims*i)] (nupts, dim, eles)

  double pvt[3] = {pivot[0], pivot[1], pivot[2]};

  //for(unsigned int d1 = 0; d1 <ndims; d1++) {
  //  pvt[d1] += offset[d1];
  //}
  //printf("pivot %lf %lf %lf\n", pivot[0], pivot[1], pivot[2]);

  double xtmp[3] = {0.0}; // fix it
  for(unsigned int d1 = 0; d1 < ndims;d1++) {
    xtmp[d1] = offset[d1];
    for(unsigned int d2 = 0; d2 < ndims; d2++){
      //xtmp[d1] += Rmat[d1*ndims+d2]*(nestedcoords_ref[nc+ncells*(d2+ndims*upt)]+offset[d2]);
      xtmp[d1] += Rmat[d1*ndims+d2]*(nestedcoords_ref[nc+ncells*(d2+ndims*upt)]-pvt[d2]);
    }
  }

  for(unsigned int d = 0; d<ndims; d++) {
    nestedcoords[nc+ncells*(d+ndims*upt)] = xtmp[d] + pvt[d];
  }
}

// for nested coords do it by element type
void move_grid_nested_wrapper(
    double* nestedcoords, 
    double* nestedcoords_ref, 
    unsigned int ncells, // total number of elements
    unsigned int npts, // points per element
    unsigned int ndims,
    double sgn,
    double* Rmat,
    double* offset,
    double* pivot,
    int stream) { 
  int threads = 128;
  int blocks = (ncells*npts+ threads - 1)/threads;

  if(stream == -1) {
    move_nested<<<blocks,threads>>>(
      nestedcoords, nestedcoords_ref, ncells, npts, ndims, sgn, Rmat, offset, pivot
    );
  } else {
    move_nested<<<blocks,threads,0,stream_handles[stream]>>>(
      nestedcoords, nestedcoords_ref, ncells, npts, ndims, sgn, Rmat, offset, pivot
    );
  }
  check_error();
}

__global__
void copy_to_mpi_rhs(
    double* base, 
    double* src,
    unsigned int* doffset, //offset from the base in char 1 byte
    unsigned int* fidx,
    unsigned int* soffset, // offset from the base in double 8 bytes
    unsigned int* nfpts,
    unsigned int* fbase,
    unsigned int nvar,
    unsigned int nface
    ) {
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid >= nface ) return;

  int fid = fidx[tid];// fid is the global id
  int doff = doffset[fid]; // in bytes
  int nft = nfpts[fid]; //
 

  int soff = soffset[tid]; // local id
  double* source = src + soff;   
  
  char* cdest = ((char*) base) + doff; 
  double* dest = (double*) cdest;// this is for first variable of the face
    
  //printf("offset for face %d\n", fbase[fid]);
  //printf("base is %ld\n", (long long) (base));
  //printf("base of face is %d %d %d %d\n", doffset[0],doffset[1], doffset[2], doffset[3]);
  for(int i=0;i<nvar;++i) { 
    double* vdest = dest + i*fbase[fid]; // fbase is offset interms of double
    for(int j=0;j<nft;++j) {
        
        //printf("source %lf \n", source[j*nvar + i]);
        
        vdest[j] = source[j*nvar + i];
    }
  }

}

// the wrapper to copy the data 
void copy_to_mpi_rhs_wrapper(
    double* base, double* src,
    unsigned int* doffset, unsigned int* fidx,  // these two decide the offset for dest
    unsigned int* soffset, // this one decide the offset for src
    unsigned int* nfpts,
    unsigned int* fbase, 
    unsigned int nvar, unsigned int nface, int stream
) {
  int threads = 256;
  int blocks = (int) (nface + threads -1) /threads;
  if(stream == -1) {
    copy_to_mpi_rhs<<<blocks, threads>>>(
        base, src, doffset, fidx, soffset, nfpts, fbase, nvar, nface
    );
  } else {
    copy_to_mpi_rhs<<<blocks, threads, 0, stream_handles[stream]>>> (
        base, src, doffset, fidx, soffset, nfpts, fbase, nvar, nface
    );
  }
}


