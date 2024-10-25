#include "codetypes.h"
#include "error.hpp"
#include "dADT.h"
#include "MeshBlock.h"
#include "funcs.hpp"

#define MAX_LEVEL 20
#define MAX_NCHECK 20
#define FTOL 1e-6f

dADT::~dADT()
{
//  clearData();
}

void dADT::clearData(void)
{
  adtInts.free_data();
  adtReals.free_data();
  adtBBox.free_data();
  coord.free_data();
}

void dADT::copyADT(ADT *adt)
{
  ndim = adt->ndim;
  nelem = adt->nelem;
  adtInts.assign(adt->adtIntegers, nelem*4);
  adtReals.assign(adt->adtReals, nelem*ndim);
  adtBBox.assign(adt->adtExtents, ndim);
  coord.assign(adt->coord, ndim*nelem);
  rrot = adt->rrot;
  if (rrot)
  {
    offset.assign(adt->offset, 3);
    Rmat.assign(adt->Rmat, 9);
    Pivot.assign(adt->Pivot,3);
  }
}

void dADT::setTransform(double *mat, double* pvt, double *off, int nDims)
{
  if (nDims != ndim/2)
    FatalError("dADT:setTransform:nDims != dADT::ndim/2");
  
  //printf("mat %ld\n", mat);
  //printf("adt mat is %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e \n",
  //  mat[0], mat[1], mat[2],
  //  mat[3], mat[4], mat[5],
  //  mat[6], mat[7], mat[8]
  //);
  rrot = true;

  //printf("pivot cpu %lf %lf %lf\n", pvt[0], pvt[1], pvt[2]);
  Rmat.assign(mat, nDims*nDims);
  Pivot.assign(pvt, nDims);
  offset.assign(off, nDims);
}

template<int nDims, int nside>
__device__
void d_searchADTstack(dADT& adt, dMeshBlock& mb, int& cellIndex, float* xsearch,
    float* rst)
{
  const int ndim = 2*nDims;
  int stack[MAX_LEVEL] = {0};
  int estack[MAX_NCHECK] = {-1};
  int size = 1;
  int ncheck = 0;

  while (size > 0)
  {
    int node = stack[size-1];
    size--;

    int ele = adt.adtInts[4*node];
    float bbox[ndim];
    for (int i = 0; i < ndim; i++)
      bbox[i] = mb.eleBBox[ndim*ele+i];

    bool flag = true;
    for (int i = 0; i < nDims; i++)
    {
      flag = (flag && (xsearch[i] >= bbox[i]-FTOL));
      flag = (flag && (xsearch[i] <= bbox[i+nDims]+FTOL));
    }

    // Insert into list of elements to check
    if (flag)
      estack[ncheck++] = ele;

    // check the left and right children now
    for (int d = 1; d < 3; d++)
    {
      int nodeChild = adt.adtInts[4*node+d];
      if (nodeChild > -1)
      {
        nodeChild = adt.adtInts[4*nodeChild+3];
        for (int i = 0; i < ndim; i++)
          bbox[i] = adt.adtReals[ndim*nodeChild+i];

        flag = true;
        for (int i = 0; i < nDims; i++)
        {
          flag = (flag && (xsearch[i] >= bbox[i]-FTOL));
          flag = (flag && (xsearch[i] <= bbox[i+nDims]+FTOL));
        }

        if (flag)
          stack[size++] = nodeChild;
      }
    }
  }

  // Perform the point/cell containment checks [avoiding thread divergence]
  for (int n = 0; n < ncheck; n++)
  {
    int ele = estack[n];
    float bbox[ndim];
    for (int i = 0; i < ndim; i++)
      bbox[i] = mb.eleBBox[ndim*ele+i];

    mb.checkContainment<nDims,nside>(ele,cellIndex,bbox,xsearch,rst);

    if (cellIndex > -1)
      return;
  }
}

template<int nDims, int nside>
__global__
void searchADT_kernel(dADT adt, dMeshBlock mb)
{
  int pt = blockDim.x * blockIdx.x + threadIdx.x;

  if (pt >= mb.nsearch) return;

  //printf("pt is %d out of total %d flag is true\n",pt,mb.nsearch);
  // check if the given point is in the bounds of the ADT
  int cellID = -1;

  float xsearch[nDims];
  for (int d = 0; d < nDims; d++) {
    xsearch[d] = mb.xsearch[nDims*pt+d];
  }

  //printf("search point x %15.7e y %15.7e z %15.7e\n", xsearch[0], xsearch[1], xsearch[2]);

  if (adt.rrot) // Transform back to ADT's coordinate system
  {
    float pivot[nDims] = {(float) adt.Pivot[0], (float) adt.Pivot[1], (float) adt.Pivot[2]};
    //float pivot[nDims] = {0.0};
    
    // update pivot 

    //for(int d1=0;d1<nDims;d1++) {
    //    pivot[d1] -= adt.offset[d1];
    //}
    
    //if(pt == 1) {
    //printf("x %f y  %f z  %f\n", xsearch[0], xsearch[1], xsearch[2]);
    //printf("px %f py %f pz %f\n", pivot[0], pivot[1], pivot[2]);
    //printf("pxi %lf pyi %lf pzi %lf\n",adt.Pivot[0], adt.Pivot[1], adt.Pivot[2]);
    //}


    float x2[nDims] = {0.0};
    for (int d1 = 0; d1 < nDims; d1++) {
      for (int d2 = 0; d2 < nDims; d2++) {
        x2[d1] += adt.Rmat[d1+nDims*d2]*(xsearch[d2]-adt.offset[d2] - pivot[d2]);
      }
    }

    for (int d = 0; d < nDims; d++) {
      xsearch[d] = x2[d] + pivot[d];
    }

    //if(pt == 1) {
    //printf("xr %f yr  %f zr  %f\n", xsearch[0], xsearch[1], xsearch[2]);
    //printf("x2 %f y2  %f z2  %f\n", x2[0], x2[1], x2[2]);
    //}
  }

  bool flag = true;
  for (int d = 0; d < nDims; d++)
  {
    flag = (flag && (xsearch[d] >= adt.adtBBox[2*d]-FTOL));
    flag = (flag && (xsearch[d] <= adt.adtBBox[2*d+1]+FTOL));
    //if (!flag && pt == 1) 
    //printf("adtbox is %lf %lf %lf %d\n", adt.adtBBox[2*d], adt.adtBBox[2*d+1], xsearch[d], d);
  }

  //printf("pt is %d out of total %d flag is %d\n",pt,mb.nsearch,flag);
  // call recursive routine to check intersections with ADT nodes
  float rst[nDims] = {0.0};
  if (flag) {
    d_searchADTstack<nDims,nside>(adt,mb,cellID,xsearch,rst);
  }

  mb.donorId[pt] = cellID;
  for (int d = 0; d < nDims; d++)
    mb.rst[nDims*pt+d] = rst[d];
}

void searchADT(dADT &adt, dMeshBlock &mb)
{
  //printf("search ADT\n");
  int threads = 32;
  int blocks = (mb.nsearch + threads - 1) / threads;

  cudaFuncSetCacheConfig(searchADT_kernel<3,2>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(searchADT_kernel<3,3>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(searchADT_kernel<3,4>, cudaFuncCachePreferL1);

  switch (mb.nvert)
  {
    case 8:
      searchADT_kernel<3,2><<<blocks, threads>>>(adt, mb); break;
    case 27:
      searchADT_kernel<3,3><<<blocks, threads>>>(adt, mb); break;
    case 64:
      searchADT_kernel<3,4><<<blocks, threads>>>(adt, mb); break;
    default:
      ThrowException("nvert case not implemented");
  }

  check_error();
}
