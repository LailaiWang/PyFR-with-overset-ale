%module(directors="1") tioga

// -----------------------------------------------------------------------------
// Header files required by any of the following C++ code
// -----------------------------------------------------------------------------
%header
%{
#include <mpi.h>
#include <stdint.h>
#include <cmath>
#include <cuda_runtime.h>
#include "tiogaInterface.h"
#include "helper.h"
%}

%include "cpointer.i"

// -----------------------------------------------------------------------------
// Header files and other declarations to be parsed as SWIG input
// -----------------------------------------------------------------------------

// SWIG interface file for MPI, and typemap for MPI_Comm to Python Comm
%include mpi4py/mpi4py.i
%mpi4py_typemap(Comm,MPI_Comm);

%pythoncallback;
// Functions declared here will be able to act like (C-style) function pointers
void tioga_dataupdate_ab(int nvar, int gradFlag);
void tioga_dataupdate_ab_send(int nvar, int gradFlag = 0);
void tioga_dataupdate_ab_recv(int nvar, int gradFlag = 0);
void tioga_preprocess_grids_(void);
void tioga_performconnectivity_(void);
void tioga_do_point_connectivity(void);
void tioga_set_transform(double *mat, double* pvt, double *off, int ndim);
void tioga_unblank_part_1(void);
void tioga_unblank_part_2(int nvar);
void tioga_set_soasz(unsigned int sz);

void tg_print_data(unsigned long long int datastart, unsigned long long int offset,
                   unsigned int nums, int dtype);
void get_nodal_basis_wrapper(int* cellIDs, double* rst, double* weights,
    double* xiGrid, int nFringe, int nSpts, int nSpts1D, int stream);

void pack_cell_coords_wrapper(
    int* ucellIDs, int* ecellIDs, 
    double* xyz, double* coord_spts, 
    unsigned int nCells,
    unsigned int nSpts, unsigned int nDims,
    unsigned int soasz, unsigned int neled2, int stream = -1);

void unpack_unblank_u_wrapper(
    int* ucellIDs, int* ecellIDs, 
    double* U_unblank, double* U_spts,
    unsigned int nCells,
    unsigned int nSpts, unsigned int nVars,
    unsigned int soasz, unsigned int neled2, int stream = -1);

void pack_fringe_coords_wrapper(
    unsigned int* fringe_fpts, double* xyz,
    double* coord_fpts, 
    int nPts, int nDims, 
    unsigned int soasz, int stream = -1);

void unpack_fringe_u_wrapper(
    double *U_fringe, double* U, 
    unsigned int* fringe_fpts,
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

void move_grid_nested_wrapper(
    double* nestedcoords, 
    double* nestedcoords_ref,
    unsigned int ncells, 
    unsigned int npts, // points per element
    unsigned int ndims,
    double sgn,
    double* Rmat,
    double* offset,
    double* pivot,
    int stream = -1); 

void move_grid_flat_wrapper(
    double* flatcoords, 
    double* flatcoords_ref,
    unsigned int npts,  // total points
    unsigned int ndims,
    double sgn,
    double* Rmat,
    double* offset,
    double* pivot,
    int stream = -1);

void reset_mpi_face_artbnd_status_wrapper(
    double* status, 
    unsigned int* mapping, 
    double val,
    unsigned int nface,
    unsigned int nfpts, unsigned int nvars, 
    unsigned int soasz, int strean=-1);

void initialize_stream_event();
void destroy_stream_event();
cudaStream_t get_stream_handle();
cudaEvent_t get_event_handle();
void sync_device();
void addrToCudaStream(unsigned long long int);
%nopythoncallback;

%ignore tioga_dataupdate_ab;
%ignore tioga_dataupdate_ab_send;
%ignore tioga_dataupdate_ab_recv;
%ignore tioga_preprocess_grids_;
%ignore tioga_performconnectivity_;
%ignore tioga_do_point_connectivity;
%ignore tioga_set_transform;
%ignore tioga_unblank_part_1;
%ignore tioga_unblank_part_2;
%ignore tioga_set_soasz;
%ignore tg_print_data;
%ignore get_nodal_basis_wrapper;
%ignore pack_cell_coords_wrapper;
%ignore unpack_unblank_u_wrapper;
%ignore unpack_fringe_u_wrapper;
%ignore unpack_fringe_grad_wrapper;
%ignore pack_fringe_coords_wrapper;
%ignore reset_mpi_face_artbnd_status_wrapper;
%ignore initialize_stream_event;
%ignore destroy_stream_event;
%ignore get_stream_handle;
%ignore get_event_handle;
%ignore move_grid_flat_wrapper;
%ignore move_grid_nested_wrapper;
%ignore sync_device;
%ignore addrToCudaStream;
%include "tiogaInterface.h"
%include "helper.h"

// define some global variables for debugging purpose
%{
int *iblank_node_tg;
int *iblank_cell_tg;
int *iblank_face_tg;
%}

// enable python callbacks by cross language polymorphism
%feature("director") callbacks;
%inline %{
struct callbacks {
  virtual void get_nodes_per_cell(int* cellid, int *nnodes)=0;
  virtual void get_nodes_per_face(int* faceid, int *nnodes)=0;
  virtual void get_receptor_nodes(int* cellid, int* nnodes, double* xyz)=0;
  virtual void get_face_nodes(int* faceid, int* nnodes, double* xyz)=0;
  virtual void donor_inclusion_test(int* cellid, double* xyz, 
                                    int* passFlag, double* rst)=0;
  virtual void convert_to_modal(int* cellid, int* nspts, double* qin,
                      int* npts, int* idx_o, double* qot)=0;
  virtual void donor_frac(int* cellid, double* xyz, int* nwts, int* inode,
                      double* wts, double* rst, int* buffsize)=0;
  virtual double get_q_spt(int cellid, int spt, int va)=0;
  virtual double get_dq_spt(int cellid, int spt, int dim, int va)=0;

  virtual unsigned long long int get_q_fpt(int faceid, int fpt, int va)=0;
  virtual unsigned long long int get_dq_fpt(int faceid, int fpt, int dim, int va)=0;
  virtual unsigned long long int get_q_spts(int& es, int& ss, int& vs , int etyp)=0;
  virtual unsigned long long int get_dq_spts(int& es, int& ss, int& vs , 
                                             int& ds, int etyp)=0;

  // for GPU 
  virtual void donor_frac_gpu(int* cellid, int, double*, double*)=0;
  virtual int get_nweights_gpu(int)=0;
  virtual void get_face_nodes_gpu(int* ids, int nf, int* nptf, double* xyz)=0;
  virtual void get_cell_nodes_gpu(int* ids, int nf, int* nptf, double* xyz)=0;
  virtual void fringe_data_to_device(int* ids, int nf, int grad, double* data)=0;
  virtual void cell_data_to_device(int* ids, int nc, int grad, double* data)=0;

  virtual unsigned  long long int get_q_spts_gpu(int& es, int& ss, int&vs, int etype)=0;
  virtual unsigned  long long int get_dq_spts_gpu(int& es, int& ss, int& vs, int& ds, int etype)=0;

  virtual ~callbacks() {}
};
%}

%feature("director") gmrespmg;
%inline %{
struct gmrespmg {
    virtual void pmgupdate()=0;
    virtual void rhsupdate()=0;
    virtual ~gmrespmg() {}
};
%}

%feature("director:except") {
  if ($error != NULL) {
    fprintf(stderr, "throw\n");
    throw Swig::DirectorMethodException();
  }
}

%exception {
  try { $action }
  catch (Swig::DirectorException &e) { fprintf(stderr, "catch\n"); SWIG_fail; }
}


%{
extern callbacks *cb_ptr;
extern gmrespmg  *pmg_ptr;
%}

%{
gmrespmg* pmg_ptr = NULL;
static void helper_pmgupdate() {
    return pmg_ptr->pmgupdate();
}
static void helper_rhsupdate() {
    return pmg_ptr->rhsupdate();
}
%}


%{
callbacks* cb_ptr = NULL;

static void helper_donor_frac_gpu(int* cellid, int n, double* loc, double* wts) {
  return cb_ptr->donor_frac_gpu(cellid,n,loc,wts);
}

static int helper_get_nweights_gpu(int cellid) {
  return cb_ptr->get_nweights_gpu(cellid);
}

static void helper_get_face_nodes_gpu(int* ids, int nf, int* nptf, double* xyz) {
  return cb_ptr->get_face_nodes_gpu(ids,nf,nptf,xyz);
}

static void helper_get_cell_nodes_gpu(int* ids, int nf, int* nptf, double* xyz) {
  return cb_ptr->get_cell_nodes_gpu(ids,nf,nptf,xyz);
}

static void helper_fringe_data_to_device(int* ids, int nf, int grad, double* data) {
  return cb_ptr->fringe_data_to_device(ids,nf,grad,data);
}

static void helper_cell_data_to_device(int* ids, int nc, int grad, double* data) {
  return cb_ptr->cell_data_to_device(ids,nc,grad,data);
}

static unsigned long long int helper_get_q_spts_gpu(int& es, int& ss, 
                                                    int& vs, int etype) {
  return cb_ptr->get_q_spts_gpu(es,ss,vs,etype);
}

static unsigned long long int helper_get_dq_spts_gpu(int& es, int& ss, int& vs,
                                                     int& ds, int etype) {
  return cb_ptr->get_dq_spts_gpu(es,ss,vs,ds,etype);
}

double* helper_array_q_gpu(int& es, int& ss, int& vs, int etyp) {
    unsigned long long int c = helper_get_q_spts_gpu(es,ss,vs,etyp);
    double *tmp = reinterpret_cast<double*> (c);
    return tmp;
}

double* helper_array_dq_gpu(int& es, int& ss, int& vs, int& ds, int etyp) {
    unsigned long long int c = helper_get_dq_spts_gpu(es,ss,vs, ds, etyp);
    double *tmp = reinterpret_cast<double*> (c);
    return tmp;
}


// for CPU
static void helper_get_nodes_per_cell(int* cellid, int* nnodes) {
  return cb_ptr->get_nodes_per_cell(cellid,nnodes);
}

static void helper_get_nodes_per_face(int* faceid, int* nnodes) {
  return cb_ptr->get_nodes_per_face(faceid,nnodes);
}

static void helper_get_receptor_nodes(int* cellid, int* nnodes, double* xyz) {
  return cb_ptr->get_receptor_nodes(cellid,nnodes, xyz);
}

static void helper_get_face_nodes(int* faceid, int* nnodes, double* xyz) {
    return cb_ptr->get_face_nodes(faceid,nnodes,xyz);
}

static void helper_donor_inclusion_test(int* cellid, double* xyz,
                                        int* passFlag, double* rst) {
    return cb_ptr->donor_inclusion_test(cellid,xyz,passFlag,rst);
}

static void helper_convert_to_modal(int* cellid, int* nspts, double* qin,
                                    int*npts, int* idxo,double*qo) {
    return cb_ptr->convert_to_modal(cellid,nspts,qin,npts,idxo,qo);
};

static void helper_donor_frac(int* cellid, double* xyz, int* nwts, 
                    int* inode, double* wts, double* rst, int* buffsize) {
    return cb_ptr->donor_frac(cellid, xyz, nwts, inode, wts, rst, buffsize);
};

static double helper_get_q_spt(int cellid, int spt, int va) {
    return cb_ptr->get_q_spt(cellid, spt, va);
}

static unsigned long long int helper_get_q_fpt(int faceid, int fpt, int va) {
    return cb_ptr->get_q_fpt(faceid,fpt,va);
}

static double helper_get_dq_spt(int cellid, int spt, int dim, int va) {
    return cb_ptr->get_dq_spt(cellid,spt,dim,va);
}

static unsigned long long int helper_get_dq_fpt(int faceid, int fpt, int dim, int va) {
    return cb_ptr->get_dq_fpt(faceid,fpt,dim,va);
}

static unsigned long long int helper_get_q_spts(int& es, int& ss, int& vs, int etyp) {
    return cb_ptr->get_q_spts(es,ss,vs,etyp);
}

static unsigned long long int helper_get_dq_spts(int& es, int& ss, int& vs,
                                                 int& ds, int etyp) {
    return cb_ptr->get_dq_spts(es,ss,vs,ds,etyp);
}

double& helper_float_qfpt(int faceid, int fpt, int va) {
    unsigned long long int c = helper_get_q_fpt(faceid,fpt,va);
    double* d = reinterpret_cast<double*> (c);
    return *d;
}

double& helper_float_dqfpt(int faceid, int fpt, int dim, int va) {
    unsigned long long int c = helper_get_dq_fpt(faceid,fpt,dim,va);
    double* d = reinterpret_cast<double*> (c);
    return *d;
}

double* helper_array_q(int& es, int& ss, int& vs, int etyp) {
    unsigned long long int c = helper_get_q_spts(es,ss,vs,etyp);
    double *tmp = reinterpret_cast<double*> (c);
    return tmp;
}

double* helper_array_dq(int& es, int& ss, int& vs, int& ds, int etyp) {
    unsigned long long int c = helper_get_dq_spts(es,ss,vs, ds, etyp);
    double *tmp = reinterpret_cast<double*> (c);
    return tmp;
}

%}

%inline %{
void tioga_set_callbacks_ptr(callbacks* cb) {
    cb_ptr = cb;
}
%}

%inline %{
void tioga_set_highorder_callback_wrapper(callbacks* cb) {
    //cb_ptr = cb;
    tioga_set_highorder_callback_(&helper_get_nodes_per_cell,
                                  &helper_get_receptor_nodes,
                                  &helper_donor_inclusion_test,
                                  &helper_donor_frac,
                                  &helper_convert_to_modal);
}
%}

%inline %{
void tioga_set_ab_callback_wrapper(callbacks* cb) {
    //cb_ptr = cb;
    tioga_set_ab_callback_(&helper_get_nodes_per_face,
                           &helper_get_face_nodes,
                           &helper_get_q_spt,
                           &helper_float_qfpt,
                           &helper_get_dq_spt,
                           &helper_float_dqfpt,
                           &helper_array_q,
                           &helper_array_dq);
}
%}

// for GPU
%inline %{
void tioga_set_ab_callback_gpu_wrapper(callbacks* cb) {
    //cb_ptr = cb;
    tioga_set_ab_callback_gpu_(&helper_fringe_data_to_device,
                           &helper_cell_data_to_device,
                           &helper_array_q_gpu,
                           &helper_array_dq_gpu,
                           &helper_get_face_nodes_gpu,
                           &helper_get_cell_nodes_gpu,
                           &helper_get_nweights_gpu,
                           &helper_donor_frac_gpu);
}
%}


%inline %{

void tioga_set_gmres_pmg_ptr () {

}

%}

// some simple helper functions
%inline %{
unsigned long long int tg_allocate_device(int maxcells, int maxupts,
                          int dim, int nvar, int vect, int itemsize) {
    long int nbytes = vect==1? dim*dim:dim;
    nbytes *= maxcells*nvar*maxupts*itemsize;
    if(itemsize == 4) {
        float* d;
        cudaMalloc((void **) &d,nbytes);
        return reinterpret_cast<unsigned long long int> (d);
    } else {
        double* d;
        cudaMalloc((void **) &d, nbytes);
        return reinterpret_cast<unsigned long long int> (d);
    }
}

unsigned long long int tg_allocate_device_int(int maxfaces) {
    int nbytes = maxfaces*sizeof(int);
    int* d;
    cudaMalloc((void **) &d, nbytes);
    
    ////////// for testing purpose ////////////////
    int *h = (int*) malloc(nbytes);
    for(int i=0;i<maxfaces;i++) h[i] = -10;
    cudaMemcpy(d,h,nbytes,cudaMemcpyHostToDevice);
    free(h);
    ///////////////////////////////////////////////

    return reinterpret_cast<unsigned long long int> (d);
}

// check the data helper


void tg_free_device(unsigned long long int d, int itemsize) {
    if(itemsize == 4) {
        float* dptr = reinterpret_cast<float*> (d);
        cudaFree(dptr);
    } else {
        double* dptr = reinterpret_cast<double*> (d);
        cudaFree(dptr);
    }
}

// copy data from host to device for double
void tg_copy_to_device(unsigned long long int a, double *data, int nbytes, int offset = 0) {
    // cast a to pointer
    double* a_d = reinterpret_cast<double* > (a);
    printf("current offset is %d\n", offset);
    cudaMemcpy(a_d, data + offset, nbytes, cudaMemcpyHostToDevice);
}

void tg_copy_to_device(unsigned long long int a, int *data, int nbytes, int offset = 0) {
    // cast a to pointer
    int* a_d = reinterpret_cast<int* > (a);
    cudaMemcpy(a_d, data + offset, nbytes, cudaMemcpyHostToDevice);
}

// copy data from host to device for float
void tg_copy_to_device(unsigned long long int a, float *data, int nbytes, int offset = 0) {
    float* a_d = reinterpret_cast<float* > (a);
    cudaMemcpy(a_d, data + offset, nbytes, cudaMemcpyHostToDevice);
}

// copy data from device to host for double
void tg_copy_to_host(unsigned long long int a, double *data, int nbytes) {
   double* a_d = reinterpret_cast<double*> (a);
   cudaMemcpy(data, a_d, nbytes, cudaMemcpyDeviceToHost);
}
// copy data from device to host for float
void tg_copy_to_host(unsigned long long int a, float *data, int nbytes) {
   float* a_d = reinterpret_cast<float*> (a);
   cudaMemcpy(data, a_d, nbytes, cudaMemcpyDeviceToHost);
}

%}

// <-- Additional C++ declations [anything that would normally go in a header]

// -----------------------------------------------------------------------------
// Additional functions which have been declared, but not defined (including
// definition in other source files which will be linked in later)
// -----------------------------------------------------------------------------

%inline
%{
// <-- Additional C++ definitions [anything that would normally go in a .cpp]
%}

// -----------------------------------------------------------------------------
// Additional Python functions to add to module
// [can use any functions/variables declared above]
// -----------------------------------------------------------------------------

%pythoncode
%{
# Python functions here
%}
