//
// This file is part of the Tioga software library
//
// Tioga  is a tool for overset grid assembly on parallel distributed systems
// Copyright (C) 2015 Jay Sitaraman
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "tioga.h"
#include "globals.h"
#include <string.h>

#include "tiogaInterface.h"

//
// All the interfaces that are 
// accessible to third party f90 and C
// flow solvers
//
//
// Jay Sitaraman
// 02/24/2014
//
extern "C" {

  void tioga_init_f90_(int *scomm)
  {
    int id_proc,nprocs;
    MPI_Comm tcomm = MPI_Comm_f2c(*scomm);
    //
    tg=new tioga();
    //
    //MPI_Comm_rank(MPI_COMM_WORLD,&id_proc);
    //MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(tcomm,&id_proc);
    MPI_Comm_size(tcomm,&nprocs);
    //
    tg->setCommunicator(tcomm,id_proc,nprocs);
    nc=NULL;
    nv=NULL;
    vconn=NULL;
  }

  void tioga_init_(MPI_Comm tcomm)
  {
    int id_proc,nprocs;

    tg = new tioga();

    MPI_Comm_rank(tcomm,&id_proc);
    MPI_Comm_size(tcomm,&nprocs);

    tg->setCommunicator(tcomm,id_proc,nprocs);
    nc=NULL;
    nv=NULL;
    vconn=NULL;
  }
  
  void tioga_registergrid_data_(int btag, int nnodes, double *xyz, int *ibl,
                                int nwbc, int nobc, int *wbcnode, int *obcnode,
                                int ntypes, int* _nv, int* _ncf, int* _nc, int **_vconn)
  {
    free(nv);
    free(nc);
    free(ncf);
    free(vconn);

    // NOTE: due to SWIG/Python wrapping, removing va_args stuff for now
    nv    = (int *) malloc(sizeof(int)*ntypes);
    nc    = (int *) malloc(sizeof(int)*ntypes);
    ncf    = (int *) malloc(sizeof(int)*ntypes);
    vconn = (int **)malloc(sizeof(int *)*ntypes);

    for (int i = 0; i < ntypes; i++)
    {
      nv[i] = _nv[i];
      ncf[i] = _ncf[i];
      nc[i] = _nc[i];
      vconn[i] = _vconn[i];
    }

    tg->registerGridData(btag,nnodes,xyz,ibl,nwbc,nobc,wbcnode,obcnode,ntypes,nv,ncf,nc,vconn);
  }

  void tioga_register_face_data_(int gtype, int *f2c, int **c2f, int *fibl, int nOverFaces,
      int nWallFaces, int nMpiFaces, int *overFaces, int *wallFaces, int *mpiFaces,
      int* mpiProcR, int* mpiFidR, int nftype, int *_nfv, int *_nf, int **_fconn)
  {
    free(nfv);
    free(nf);
    free(fconn);

    nfv   = (int *) malloc(sizeof(int)*nftype);
    nf    = (int *) malloc(sizeof(int)*nftype);
    fconn = (int **)malloc(sizeof(int *)*nftype);

    for (int i = 0; i < nftype; i++)
    {
      nfv[i] = _nfv[i];
      nf[i] = _nf[i];
      fconn[i] = _fconn[i];
    }

    tg->registerFaceConnectivity(gtype, nftype, nf, nfv, fconn, f2c, c2f, fibl,
        nOverFaces, nWallFaces, nMpiFaces, overFaces, wallFaces, mpiFaces,
        mpiProcR, mpiFidR);
  }

  void tioga_register_amr_global_data_(int nf, int qstride, double *qnodein,
				      int *idata,double *rdata,
				      int ngridsin,int qnodesize)
  {
    /*
    printf("A:%d %d %d %d\n",*nf,*qstride,*ngridsin,*qnodesize);
    printf("B:%d %d %d %d\n",nf[0],qstride[0],ngridsin[0],qnodesize[0]);
    printf("C:%d %d %d %d\n",idata[12],idata[13],idata[14],idata[15]);
    tracei(*qstride);
    tracei(*ngridsin);
    tracei(*qnodesize);
    */
    tg->register_amr_global_data(nf,qstride,qnodein,idata,rdata,ngridsin,qnodesize);
  }


  void tioga_register_amr_patch_count_(int npatches)
  {
    tg->set_amr_patch_count(npatches);
  }

  void tioga_register_amr_local_data_(int ipatch,int global_id,int *iblank,double *q)
  {
    tg->register_amr_local_data(ipatch,global_id,iblank,q);
  }

  double* tioga_get_igbp_list(void)
  {
    double* igbp_ptr = NULL;
    tg->get_igbp_ptr(igbp_ptr);
    return igbp_ptr;
  }

  int tioga_get_n_igbps(void)
  {
    return tg->get_n_igbps();
  }

  void tioga_preprocess_grids_(void)
  {
    tg->profile();
  }

  void tioga_performconnectivity_(void)
  {
    tg->performConnectivity();
  }

  void tioga_performconnectivity_highorder_(void)
  {
    tg->performConnectivityHighOrder();
  }

  void tioga_performconnectivity_amr_(void)
  {
   tg->performConnectivityAMR();
  }
  void tioga_dataupdate_amr(double **q, int nvar, int interptype)
  {
    tg->dataUpdate_AMR(nvar,q,interptype);
  }
  void tioga_dataupdate_(double *q,int *nvar,char *itype)
  {
    int interptype;
    if (strstr(itype,"row")) 
      {
	interptype=0;
      }
    else if (strstr(itype,"column")) 
      {
	interptype=1;
      }
    else
      {
	printf("#tiogaInterface.C:dataupdate_:unknown data orientation\n");
	return;
      }
    if (tg->ihighGlobal==0) 
      {
	if (tg->iamrGlobal==0) 
	  {
	    tg->dataUpdate(*nvar,q,interptype);
	  }
	else
	  {
	    tg->dataUpdate_AMR(*nvar,&q,interptype);
	  }
      }
    else
      {
	if (tg->iamrGlobal==0) 
	  {
	    tg->dataUpdate_highorder(*nvar,q,interptype);
	  }
	else
	  {
	    printf("Data udpate between high-order near-body and AMR cartesian Not implemented yet\n");
	  }
      }
  }

  void tioga_dataupdate_ab(int nvar, int gradFlag)
  {
    tg->dataUpdate_artBnd(nvar, gradFlag);
  }

  void tioga_dataupdate_ab_send(int nvar, int gradFlag)
  {
    tg->dataUpdate_artBnd_send(nvar, gradFlag);
  }

  void tioga_dataupdate_ab_recv(int nvar, int gradFlag)
  {
    tg->dataUpdate_artBnd_recv(nvar, gradFlag);
  }

  void tioga_writeoutputfiles_(double *q,int *nvar,char *itype)
  {
    int interptype;
    if (strstr(itype,"row")) 
      {
	interptype=0;
      }
    else if (strstr(itype,"column")) 
      {
	interptype=1;
      }
    else
      {
	printf("#tiogaInterface.C:dataupdate_:unknown data orientation\n");
	return;
      }
    tg->writeData(*nvar,q,interptype);
  }    
  void tioga_getdonorcount_(int *dcount,int *fcount)
  {
    tg->getDonorCount(dcount,fcount);
  }
  void tioga_getdonorinfo_(int *receptors,int *indices,double *frac,int *dcount)
  {
    tg->getDonorInfo(receptors,indices,frac,dcount);
  }

  void tioga_setsymmetry_(int *isym)
  {
    tg->setSymmetry(*isym);
  }

  void tioga_setresolutions_(double *nres,double *cres)
  {
    tg->setResolutions(nres,cres);
  }
  
  void tioga_setcelliblank_(int *iblank_cell)
  {
    tg->set_cell_iblank(iblank_cell);
  }

  void tioga_set_highorder_callback_(void (*f1)(int*, int*),
				    void (*f2)(int *,int *,double *),
				    void (*f3)(int *,double *,int *,double *),
				    void (*f4)(int *,double *,int *,int *,double *,double *,int *),
				     void (*f5)(int *,int *,double *,int *,int *,double *))
  {
    tg->setcallback(f1,f2,f3,f4,f5);
    //get_nodes_per_cell=f1;
    //get_receptor_nodes=f2;
    //donor_inclusion_test=f3;
    //donor_frac=f4;
    //convert_to_modal=f5;
  }

  void tioga_set_p4est_(void)
  {
    tg->set_p4est();
  }

  void tioga_set_p4est_search_callback_(void (*f1)(double *xsearch,int *process_id,int *cell_id,int *npts),
          void (*f2)(int *pid,int *iflag))
  {
    tg->setp4estcallback(f1,f2);
  //jayfixme  tg->set_p4est_search_callback(f1);
  }

  void tioga_set_ab_callback_(void (*gnf)(int* id, int* npf),
                              void (*gfn)(int* id, int* npf, double* xyz),
                              double (*gqs)(int ic, int spt, int var),
                              double& (*gqf)(int ff, int fpt, int var),
                              double (*ggs)(int ic, int spt, int dim, int var),
                              double& (*ggf)(int ff, int fpt, int dim, int var),
                              double* (*gqss)(int& es, int& ss, int& vs, int etype),
                              double* (*gdqs)(int& es, int& ss, int& vs, int& ds, int etype))
  {
    tg->set_ab_callback(gnf, gfn, gqs, gqf, ggs, ggf, gqss, gdqs);
  }

  void tioga_set_ab_callback_gpu_(void (*h2df)(int* ids, int nf, int grad, double *data),
                                  void (*h2dc)(int* ids, int nc, int grad, double *data),
                                  double* (*gqd)(int etype),
                                  double* (*gdqd)(int& es, int& ss, int& vs, int& ds, int etype),
                                  void (*gfng)(int*, int, int*, double*),
                                  void (*gcng)(int*, int, int*, double*),
                                  int (*gnw)(int),
                                  void (*dfg)(int*, int, double*, double*))
  {
    tg->set_ab_callback_gpu(h2df,h2dc,gqd,gdqd,gfng,gcng,gnw,dfg);
  }

  void tioga_register_moving_grid_data(double* grid_vel, double* offset, 
                                       double* Rmat, double* Pivot)
  {
    tg->registerMovingGridData(grid_vel, offset, Rmat, Pivot);
  }

  void tioga_set_amr_callback_(void (*f1)(int *,double *,int *,double *))
  {
    tg->set_amr_callback(f1);
  }

  void tioga_set_transform(double *rmat, double *pvt,  double *offset, int ndim)
  {
    tg->setTransform(rmat, pvt, offset, ndim);
  }
  
  void tioga_do_point_connectivity(void)
  {
    tg->doPointConnectivity();
  }

  void tioga_unblank_part_1(void)
  {
    tg->unblankPart1();
  }

  void tioga_unblank_part_2(int nvar)
  {
    tg->unblankPart2(nvar);
  }

  void tioga_unblank_all_grids(int nvar)
  {
    tg->unblankAllGrids(nvar);
  }


  callbackFuncs tioga_get_callbacks(void)
  {
    callbackFuncs cbs;
    cbs.setTransform = tioga_set_transform;
    return cbs;
  }

  void tioga_delete_(void)
   {
    delete tg;
    free(nc);
    free(nv);
    free(vconn);
    free(nf);
    free(nfv);
    free(fconn);
   }

  void tioga_set_stream_handle(void* stream, void* event)
  {
#ifdef _GPU
    // Using void*'s for the sake of a build-independent wrapping interface
    tg->set_stream_handle(*(cudaStream_t*)stream, *(cudaEvent_t*)event);
#endif
  }

  void tioga_set_device_geo_data(double* xyz, double* coord, int* ibc, int* ibf)
  {
#ifdef _GPU
    tg->registerDeviceGridData(xyz, coord, ibc,  ibf);
#endif
  }

  void tioga_set_soasz(unsigned int sz) {
    tg->set_soasz(sz);
  }

  void tioga_set_maxnface_maxnfpts(unsigned int maxnface, unsigned int maxnfpts) {
    tg->set_maxnface_maxnfpts(maxnface, maxnfpts);
  }
  
  void tioga_set_face_numbers(unsigned int nmpif, unsigned int nbcf) {
    tg->set_face_numbers(nmpif, nbcf);
  }

  void tioga_set_face_fpts(unsigned long long ffpts, unsigned int ntface) {
    int* fptr = reinterpret_cast<int*>(ffpts);
    tg->set_face_fpts(fptr, ntface);
  }
  
  void tioga_set_fcelltypes(unsigned long long fctype, unsigned int ntface) {
    int* cptr = reinterpret_cast<int*>(fctype);
    tg->set_fcelltypes(cptr, ntface);
  }

  void tioga_set_fposition(unsigned long long fpos, unsigned int ntface) {
    int* posptr = reinterpret_cast<int*>(fpos);
    tg->set_fposition(posptr, ntface);
  }

  void tioga_set_interior_mapping(unsigned long long basedata, 
                                  unsigned long long grad_basedata,
                                  unsigned long long faddr,
                                  unsigned long long maddr, 
                                  unsigned long long gmaddr,
                                  unsigned long long gsaddr,
                                  int nfpts) {
    int* faceinfo = reinterpret_cast<int*>(faddr);
    int* mapping = reinterpret_cast<int*>(maddr);
    int* grad_mapping = reinterpret_cast<int*>(gmaddr);
    int* grad_strides = reinterpret_cast<int*>(gsaddr);
    tg->set_interior_mapping(basedata,grad_basedata,faceinfo,mapping,grad_mapping,grad_strides,nfpts);
  }

  void tioga_figure_out_interior_artbnd_target(unsigned long long faddr, unsigned int nfringe) {
    int* fringe = reinterpret_cast<int*>(faddr);
    tg->figure_out_interior_artbnd_target(fringe, nfringe);
  }

  void tioga_set_mpi_mapping(unsigned long long basedata,
                             unsigned long long faddr,
                             unsigned long long maddr, int nfpts) {
    int* faceinfo = reinterpret_cast<int*>(faddr);
    int* mapping = reinterpret_cast<int*>(maddr);
    tg->set_mpi_mapping(basedata, faceinfo, mapping, nfpts);
  }

  void tioga_set_mpi_rhs_mapping(unsigned long long basedata,
                                 unsigned long long maddr,
                                 unsigned long long saddr, int nfpts) {
    long long int* mapping = reinterpret_cast<long long int*>(maddr);
    int* strides = reinterpret_cast<int*>(saddr);
    tg->set_mpi_rhs_mapping(basedata, mapping, strides, nfpts);
  }

  void tioga_figure_out_mpi_artbnd_target(unsigned long long faddr, unsigned int nfringe) {
    int* fringe = reinterpret_cast<int*>(faddr);
    tg->figure_out_mpi_artbnd_target(fringe, nfringe);
  }
  
  void tioga_set_data_reorder_map(unsigned long long saddr, unsigned long long uaddr, unsigned int ncells) {
    int* srted = reinterpret_cast<int*>(saddr);
    int* unsrted = reinterpret_cast<int*>(uaddr);
    tg->set_data_reorder_map(srted, unsrted, ncells);
  }

  void tioga_set_bc_rhs_basedata(unsigned long long int basedata) {
    tg->set_overset_rhs_basedata(basedata);
  }
  
  void tioga_set_bc_mapping(unsigned long long basedata,
                             unsigned long long faddr,
                             unsigned long long maddr, int nfpts) {
    int* faceinfo = reinterpret_cast<int*>(faddr);
    int* mapping = reinterpret_cast<int*>(maddr);
    tg->set_overset_mapping(basedata, faceinfo, mapping, nfpts);
  }

  void tioga_figure_out_bc_artbnd_target(unsigned long long faddr, unsigned int nfringe) {
    int* fringe = reinterpret_cast<int*>(faddr);
    tg->figure_out_overset_artbnd_target(fringe, nfringe);
  }

  void tioga_update_fringe_face_info(unsigned int flag) {
    tg->update_fringe_face_info(flag);
  }
 
  void tioga_reset_mpi_face_artbnd_status_pointwise(unsigned int nvar) {
    tg->reset_mpi_face_artbnd_status_pointwise(nvar);
  }
 
  void tioga_reset_entire_mpi_face_artbnd_status_pointwise(unsigned int nvar) {
    tg->reset_entire_mpi_face_artbnd_status_pointwise(nvar);
  } 

  void tioga_prepare_interior_artbnd_target_data(double* data, int nvar) {
    tg->prepare_interior_artbnd_target_data(data, nvar);
  }

  void tioga_prepare_interior_artbnd_target_data_gradient(double* data, int nvar, int dim) {
    tg->prepare_interior_artbnd_target_data_gradient(data, nvar, dim);
  }

  void tioga_prepare_overset_artbnd_target_data(double* data, int nvar) {
    tg->prepare_overset_artbnd_target_data(data, nvar);
  }
 
  void tioga_prepare_mpi_artbnd_target_data(double* data, int nvar) {
    tg->prepare_mpi_artbnd_target_data(data, nvar);
  }
  
  void tioga_set_facecoords_mapping(unsigned long long int base, unsigned long long int faddr, unsigned long long int maddr, int nfpts) {
    int* faceinfo = reinterpret_cast<int*>(faddr);
    int* mapping = reinterpret_cast<int*>(maddr);
    tg->set_facecoords_mapping(base, faceinfo, mapping, nfpts);
  }

  void tioga_set_cell_info_by_type(unsigned int nctypes, unsigned int ncells,
        unsigned long long caddr, unsigned long long nuptsaddr, 
        unsigned long long uaddr, unsigned long long duaddr, 
        unsigned long long baseaddr,
        unsigned long long csaddr, unsigned long long cbaseaddr
    ) {
    int* celltypes = reinterpret_cast<int*>(caddr);
    int* nupts_per_type = reinterpret_cast<int*>(nuptsaddr);
    int* ustrides = reinterpret_cast<int*>(uaddr);
    int* dustrides = reinterpret_cast<int*>(duaddr);
    unsigned long long* du_basedata = reinterpret_cast<unsigned long long*>(baseaddr);
    int* cstrides = reinterpret_cast<int*>(csaddr);
    unsigned long long* c_basedata = reinterpret_cast<unsigned long long*>(cbaseaddr);
    tg->set_cell_info_by_type(nctypes, ncells, celltypes, nupts_per_type, ustrides, dustrides, du_basedata, cstrides, c_basedata);
  }

  void tioga_set_solution_points(unsigned long long taddr, unsigned long long caddr, unsigned long long daddr) {
    int* types = reinterpret_cast<int*>(taddr);
    int* cnupts = reinterpret_cast<int*>(caddr);
    double* data = reinterpret_cast<double*>(daddr);
    tg->set_solution_points(types, cnupts, data);
  }
}
