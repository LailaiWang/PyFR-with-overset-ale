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
#include "MeshBlock.h"
#include "superMesh.hpp"
#include "utils.h"
#include "math_funcs.h"
#include "linklist.h"

using namespace tg_funcs;

void MeshBlock::setData(int btag,int nnodesi,double *xyzi, int *ibli,int nwbci,
    int nobci,int *wbcnodei,int *obcnodei,int ntypesi,int *nvi,int *ncfi,
    int *nci,int **vconni)
{
  //
  // set internal pointers
  //
  meshtag=btag;
  nnodes=nnodesi;
  x=xyzi;
  iblank=ibli;
  nwbc=nwbci;
  nobc=nobci;
  wbcnode=wbcnodei;
  obcnode=obcnodei;
  //
  ntypes=ntypesi;
  //
  nv=nvi;
  nc=nci;
  ncf=ncfi;
  vconn=vconni;
  //
  //tracei(nnodes);
  //for (int i = 0; i < ntypes; i++) tracei(nc[i]);
  ncells=0;
  for (int i = 0; i < ntypes; i++) ncells+=nc[i];
}

void MeshBlock::setFaceData(int _gtype, int _nftype, int *_nf, int *_nfv,
    int **_f2v, int *_f2c, int **_c2f, int *_ib_face, int nOver, int nWall, int nMpi,
    int *oFaces, int *wFaces, int *mFaces, int *procR, int *idR)
{
  gridType = _gtype;
  nftype = _nftype;
  nf = _nf;
  nfv = _nfv;
  fconn = _f2v;
  f2c = _f2c;
  c2f = _c2f;
  iblank_face = _ib_face;
  nOverFaces = nOver;
  nWallFaces = nWall;
  nMpiFaces = nMpi;
  overFaces = oFaces;
  wallFaces = wFaces;
  mpiFaces = mFaces;
  mpiProcR = procR;
  mpiFidR = idR;

  nfaces = 0;
  for (int i = 0; i < nftype; i++)
    nfaces += nf[i];
}

#ifdef _GPU
void MeshBlock::setDeviceData(double* xyz, double* coords, int* ibc, int* ibf)
{
  mb_d.setDeviceData(xyz,coords,ibc,ibf);
}

#endif

void MeshBlock::preprocess(void)
{
  // set all iblanks = 1
  for (int i = 0; i < nnodes; i++) iblank[i] = NORMAL;

  // find oriented bounding boxes
  free(obb);

  obb = (OBB *) malloc(sizeof(OBB));

  findOBB(x,obb->xc,obb->dxc,obb->vec,nnodes);

  tagBoundary();
}

void MeshBlock::updateOBB(void)
{
  free(obb);
  obb = (OBB *) malloc(sizeof(OBB));

  findOBB(x,obb->xc,obb->dxc,obb->vec,nnodes);
}

/** Calculate 'cellRes' / 'nodeRes' (cell volume) for each cell / node
 *  and tag overset-boundary nodes by setting their nodeRes to a big value
 */
void MeshBlock::tagBoundary(void)
{
  std::vector<int> inode;
  double xv[8][3];
  std::vector<double> xv2;
  std::vector<int> iflag(nnodes, 0);

  // Do this only once
  // i.e. when the meshblock is first initialized, cellRes would be NULL in
  // this case
  cellRes.resize(ncells);
  nodeRes.resize(nnodes);

  if (userSpecifiedNodeRes == NULL && userSpecifiedCellRes == NULL)
  {
    for (int i = 0; i < nnodes; i++) nodeRes[i] = 0.0;

    int k = 0;
    // loop over different element types
    for (int n = 0; n < ntypes; n++)
    {
      int nvert = nv[n];
      inode.resize(nvert);
      if (nvert > 8) xv2.resize(nvert*3);

      nvert = nNodesToFirstOrder(ncf[n], nv[n]);

      for (int i = 0; i < nc[n]; i++)
      {
        double vol = 0.;
          for (int m = 0; m < nv[n]; m++)
          {
            inode[m] = vconn[n][nv[n]*i+m]-BASE;
            if (m < nvert)
            {
              int i3 = 3*inode[m];
              for (int j = 0; j < 3; j++)
                xv[m][j] = x[i3+j];
            }
          }
          //vol = computeVolume(&xv[0][0], nvert, 3);
          vol = computeCellVolume(xv, nvert);
//        }
        cellRes[k++] = vol*resolutionScale;
        for (int m = 0; m < nv[n]; m++)
        {
          iflag[inode[m]]++;
          nodeRes[inode[m]] += vol*resolutionScale;
        }
      }
    }
  }
  else
  {
    int k = 0;
    for (int n = 0; n < ntypes; n++)
    {
      for (int i = 0; i < nc[n]; i++)
      {
        cellRes[k] = userSpecifiedCellRes[k];
        k++;
      }
    }
    for (int k = 0; k < nnodes; k++) nodeRes[k] = userSpecifiedNodeRes[k];
  }

  // compute nodal resolution as the average of all the cells associated with it.
  // This takes care of partition boundaries as well.
  for (int i = 0; i < nnodes; i++)
  {
    if (iflag[i] != 0)
      nodeRes[i] /= iflag[i];
    iflag[i] = 0;
  }

  // Compute DS for overset boundary nodes
  obcRes.resize(nobc);
  for (int i = 0; i < nobc; i++)
    obcRes[i] = std::cbrt(nodeRes[obcnode[i]]);

  // now tag the boundary nodes
  // reuse the iflag array
  if (iartbnd)
  {
    for (int i = 0; i < nobc; i++)
      nodeRes[obcnode[i]-BASE] = BIGVALUE;
  }
  else
  {
    for (int i = 0; i < nobc; i++)
    {
      iflag[(obcnode[i]-BASE)] = 1;
    }

    // now tag all the nodes of boundary cells
    // to be mandatory receptors
    for (int n = 0; n < ntypes; n++)
    {
      int nvert = nv[n];
      inode.resize(nvert);
      for (int i = 0; i < nc[n]; i++)
      {
        int itag = 0;
        for (int m = 0; m < nvert; m++)
        {
          inode[m] = vconn[n][nvert*i+m]-BASE;
          if (iflag[inode[m]]) itag = 1;
        }
        if (itag)
        {
          for (int m = 0; m < nvert; m++)
          {
            nodeRes[inode[m]]=BIGVALUE;
          }
        }
      }
    }

    // now tag all the cells which have mandatory receptors as nodes as not
    // acceptable donors
    int k = 0;
    for (int n = 0; n < ntypes; n++)
    {
      int nvert = nv[n];
      inode.resize(nvert);
      for (int i = 0; i < nc[n]; i++)
      {
        for (int m = 0; m < nvert; m++)
        {
          inode[m] = vconn[n][nvert*i+m]-BASE;
          if (nodeRes[inode[m]] == BIGVALUE) //iflag[inode[m]])
          {
            cellRes[k]=BIGVALUE;
            break;
          }
        }
        k++;
      }
    }
  }
}

void MeshBlock::setupADT(void)
{
  for (int d = 0; d < 3; d++)
  {
    aabb[d]   =  BIG_DOUBLE;
    aabb[d+3] = -BIG_DOUBLE;
  }

  /// TODO: take in a bounding box of the region we're interested in searching (search.c)
  elementBbox.resize(ncells*6);
  elementList.resize(ncells);

  double xmin[3], xmax[3];
  for (int i = 0; i < ncells; i++)
  {
    int isum = 0;
    int ic = i;
    int n;
    for (n = 0; n < ntypes; n++)
    {
      isum += nc[n];
      if (ic < isum)
      {
        ic = i - (isum - nc[n]);
        break;
      }
    }

    int nvert = nv[n];
    xmin[0] = xmin[1] = xmin[2] =  BIGVALUE;
    xmax[0] = xmax[1] = xmax[2] = -BIGVALUE;
    for (int m = 0; m < nvert; m++)
    {
      int i3 = 3*(vconn[n][nvert*ic+m]-BASE);
      for (int j = 0; j < 3; j++)
      {
        xmin[j] = min(xmin[j], x[i3+j]);
        xmax[j] = max(xmax[j], x[i3+j]);
        // Overall partition bounding box
        aabb[j]   = min(aabb[j], x[i3+j]);
        aabb[j+3] = max(aabb[j+3], x[i3+j]);
      }
    }

    elementBbox[6*i] = xmin[0];
    elementBbox[6*i+1] = xmin[1];
    elementBbox[6*i+2] = xmin[2];
    elementBbox[6*i+3] = xmax[0];
    elementBbox[6*i+4] = xmax[1];
    elementBbox[6*i+5] = xmax[2];

    elementList[i] = i;
  }

  if (adt)
  {
    adt->clearData();
  }
  else
  {
    adt=new ADT[1];
  }

  adt->buildADT(2*nDims, ncells, elementBbox.data());

#ifdef _GPU
  ncells_adt = ncells;
  mb_d.dataToDevice(nDims,nnodes,ncells,ncells_adt,nsearch,nv,nc,elementList.data(),
                    elementBbox.data(),isearch.data(),xsearch.data(),myid);

  adt_d.copyADT(adt);

  /* ---- Direct Cut Setup ---- */
  // disabled
  mb_d.extraDataToDevice(vconn[0]);
#endif
}


void MeshBlock::rebuildADT(void)
{
  /// TODO: make sure we're including all possible future donors at boundaries where other grids may move in
  /// [Include any elements assigned to 'DC_CUT' from hole cutting]?

  std::set<int> donorEles, adtEles;

  if (haveDonors)
  {
    // Collect all current donor elements & their nearest neighbors
    for (int i = 0; i < donorId.size(); i++)
    {
      int ic = donorId[i];
      donorEles.insert(ic);
    }

    for (auto ic : donorEles)
    {
      if (ic < 0) continue;

      int n;
      int icBT = ic;
      for (n = 0; n < ntypes; n++)
      {
        icBT -= nc[n];
        if (icBT > 0) continue;

        icBT += nc[n];
        break;
      }

      int nface = ncf[n];
      adtEles.insert(ic);
      for (int j = 0; j < nface; j++)
      {
        adtEles.insert(c2c[n][nface*icBT+j]);
      }
    }

    // Also add in elements from boundaries [incl. overset/cut boundaries]
    for (int i = 0; i < nreceptorFaces; i++)
    {
      int ff = ftag[i];
      int ic1 = f2c[2*ff+0];
      int ic2 = f2c[2*ff+1];
      adtEles.insert(ic1);
      adtEles.insert(ic2);
    }

    for (int i = 0; i < nMpiFaces; i++)
    {
      int ff = mpiFaces[i];
      int ic = f2c[2*ff+0];
      adtEles.insert(ic);
    }
  }
  else
  {
    // We're probably still in initialization - just add all elements [Need to search full grid at least once]
    for (int i = 0; i < ncells; i++)
      adtEles.insert(i);
  }

  adtEles.erase(-1);

  // Remove eles which don't overlap with the search point oriented bounding box

  OBB obq;
  findOBB(xsearch.data(),obq.xc,obq.dxc,obq.vec,nsearch);

  std::set<int> newAdtEles; // = adtEles;

  for (auto ic : adtEles)
  {
    int nvert = nv[0];

    double xmin[3];
    double xmax[3];
    double xd[3], dxc[3];
    xmin[0] = xmin[1] = xmin[2] =  BIGVALUE;
    xmax[0] = xmax[1] = xmax[2] = -BIGVALUE;
    for (int m = 0; m < nvert; m++)
    {
      int i3 = 3*(vconn[0][nvert*ic+m]-BASE); /// TODO: ncelltypes
      for (int j = 0; j < 3; j++)
      {
        xd[j] = 0;
        for (int k = 0; k < 3; k++)
          xd[j] += (x[i3+k]-obq.xc[k])*obq.vec[j][k];
        xmin[j] = min(xmin[j],xd[j]);
        xmax[j] = max(xmax[j],xd[j]);
      }
      for (int j = 0; j < 3; j++)
      {
        xd[j] = (xmax[j]+xmin[j])*0.5;
        dxc[j] = (xmax[j]-xmin[j])*0.5;
      }
    }

    if (fabs(xd[0]) <= (dxc[0]+obq.dxc[0]) &&
        fabs(xd[1]) <= (dxc[1]+obq.dxc[1]) &&
        fabs(xd[2]) <= (dxc[2]+obq.dxc[2]))
    {
      newAdtEles.insert(ic);
    }
  }

  adtEles = newAdtEles;
  ncells_adt = adtEles.size();

  elementList.resize(ncells_adt);
  elementBbox.resize(ncells_adt*6);

  int I = 0;
  for (auto ic : adtEles)
  {
    elementList[I] = ic;
    I++;
  }

  double xmin[3], xmax[3];
  for (int i = 0; i < ncells_adt; i++)
  {
    int isum = 0;
    int ic = elementList[i];
    int n;
    for (n = 0; n < ntypes; n++)
    {
      isum += nc[n];
      if (ic < isum)
      {
        ic = ic - (isum - nc[n]);
        break;
      }
    }

    int nvert = nv[n];
    xmin[0] = xmin[1] = xmin[2] =  BIGVALUE;
    xmax[0] = xmax[1] = xmax[2] = -BIGVALUE;
    for (int m = 0; m < nvert; m++)
    {
      int i3 = 3*(vconn[n][nvert*ic+m]-BASE);
      for (int j = 0; j < 3; j++)
      {
        xmin[j] = min(xmin[j], x[i3+j]);
        xmax[j] = max(xmax[j], x[i3+j]);
      }
    }

    elementBbox[6*i] = xmin[0];
    elementBbox[6*i+1] = xmin[1];
    elementBbox[6*i+2] = xmin[2];
    elementBbox[6*i+3] = xmax[0];
    elementBbox[6*i+4] = xmax[1];
    elementBbox[6*i+5] = xmax[2];
  }

  if (adt)
  {
    adt->clearData();
  }
  else
  {
    adt=new ADT[1];
  }

  adt->buildADT(2*nDims, ncells_adt, elementBbox.data());

#ifdef _GPU
  mb_d.updateADTData(ncells_adt,elementList.data(),elementBbox.data());

  adt_d.copyADT(adt);
#endif
}

void MeshBlock::getIgbpData(double *& igbp_ptr)
{
  // Setup the Inter-Grid Boundary Point (IGBP) data list
  std::vector<double> loc_igbpdata(nobc*4);

  for (int i = 0; i < nobc; i++)
  {
    const int iv = obcnode[i];
    loc_igbpdata[4*i  ] = x[3*iv  ];
    loc_igbpdata[4*i+1] = x[3*iv+1];
    loc_igbpdata[4*i+2] = x[3*iv+2];
    loc_igbpdata[4*i+3] = obcRes[i];
  }

  // Get the number of obc nodes on each processor (for later communication)
  std::vector<int> nobc_proc(nproc);
  MPI_Allgather(&nobc,1,MPI_INT,nobc_proc.data(),1,MPI_INT,MPI_COMM_WORLD);

  // Counts & Offset data needed for Allgatherv
  std::vector<int> recvCnts(nproc);
  std::vector<int> recvDisp(nproc);
  for (int i = 0; i < nproc; i++)
  {
    recvCnts[i] = nobc_proc[i]*4;
    if (i > 0)
      recvDisp[i] = recvDisp[i-1] + recvCnts[i-1];
  }

  // Global array of IGBP data
  int nobc_g = 0;
  for (int i = 0; i < nproc; i++)
    nobc_g += nobc_proc[i];

  igbpdata.resize(nobc_g*4);

  // Get obc nodes from all ranks
  MPI_Allgatherv(loc_igbpdata.data(), nobc*4, MPI_DOUBLE, igbpdata.data(), recvCnts.data(), recvDisp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  igbp_ptr = igbpdata.data();
}

void MeshBlock::writeGridFile(int bid)
{
  char fname[80];
  char intstring[7];
  char hash,c;
  int i,n,j;
  int bodytag;
  FILE *fp;
  int ba;
  int nvert;

  sprintf(intstring,"%d",100000+bid);
  sprintf(fname,"part%s.dat",&(intstring[1]));
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"Tioga output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"IBLANK\"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEPOINT\n",nnodes,
	  ncells);
  for (int i = 0; i < nnodes; i++)
    {
      fprintf(fp,"%.14e %.14e %.14e %d\n",x[3*i],x[3*i+1],x[3*i+2],iblank[i]);
    }

  ba=1-BASE;
  for (int n = 0; n < ntypes; n++)
    {
      nvert=nv[n];
      for (int i = 0; i < nc[n]; i++)
	{
	  if (nvert==4)
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba);
	    }
	  else if (nvert==5) 
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba);
	    }
	  else if (nvert==6) 
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+5]+ba,
		      vconn[n][nvert*i+5]+ba);
	    }
	  else if (nvert==8)
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+5]+ba,
		      vconn[n][nvert*i+6]+ba,
		      vconn[n][nvert*i+7]+ba);
	    }
	}
    }
  fclose(fp);
  return;
}

void MeshBlock::writeCellFile(int bid, int* flag)
{
  char fname[80];
  char qstr[2];
  char intstring[7];
  char hash,c;
  int i,n,j;
  int bodytag;
  FILE *fp;
  int ba;
  int nvert;

  sprintf(intstring,"%d",100000+bid);
  sprintf(fname,"cell%s.dat",&(intstring[1]));
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"Tioga output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"IBLANK\",\"IBLANK_CELL\" ");
  fprintf(fp,"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEBLOCK\n",nnodes,
	  ncells);
  fprintf(fp,"VARLOCATION =  (1=NODAL, 2=NODAL, 3=NODAL, 4=NODAL,5=CELLCENTERED)\n");
  for (int i = 0; i < nnodes; i++) fprintf(fp,"%lf\n",x[3*i]);
  for (int i = 0; i < nnodes; i++) fprintf(fp,"%lf\n",x[3*i+1]);
  for (int i = 0; i < nnodes; i++) fprintf(fp,"%lf\n",x[3*i+2]);
  for (int i = 0; i < nnodes; i++) fprintf(fp,"%d\n",iblank[i]);
  if (flag != NULL)
    for (int i = 0; i < ncells; i++) fprintf(fp,"%d\n",flag[i]);
  else
    for (int i = 0; i < ncells; i++) fprintf(fp,"%d\n",iblank_cell[i]);
  ba=1-BASE;
  for (int n = 0; n < ntypes; n++)
    {
      nvert=nv[n];
      for (int i = 0; i < nc[n]; i++)
	{
	  if (nvert==4)
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba);
	    }
	  else if (nvert==5) 
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba);
	    }
	  else if (nvert==6) 
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+5]+ba,
		      vconn[n][nvert*i+5]+ba);
	    }
    else if (nvert>=8)
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+5]+ba,
		      vconn[n][nvert*i+6]+ba,
		      vconn[n][nvert*i+7]+ba);
	    }
	}
    }
  fclose(fp);
  return;
}

void MeshBlock::writeFlowFile(int bid,double *q,int nvar,int type)
{
  char fname[80];
  char qstr[2];
  char intstring[7];
  char hash,c;
  int i,n,j;
  int bodytag;
  FILE *fp;
  int ba;
  int nvert;

  sprintf(intstring,"%d",100000+bid);
  sprintf(fname,"flow%s.dat",&(intstring[1]));
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"Tioga output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"IBLANK\" ");
  for (int i = 0; i < nvar; i++)
    {
      sprintf(qstr,"Q%d",i);
      fprintf(fp,"\"%s\",",qstr);
    }
  fprintf(fp,"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEPOINT\n",nnodes,
	  ncells);

  if (type==0)
    {
      for (int i = 0; i < nnodes; i++)
	{
	  fprintf(fp,"%lf %lf %lf %d ",x[3*i],x[3*i+1],x[3*i+2],iblank[i]);
	  for (int j = 0; j < nvar; j++)
	    fprintf(fp,"%lf ",q[i*nvar+j]);      
	  //for (int j = 0; j < nvar; j++)
	  //  fprintf(fp,"%lf ", x[3*i]+x[3*i+1]+x[3*i+2]);
          fprintf(fp,"\n");
	}
    }
  else
    {
      for (int i = 0; i < nnodes; i++)
        {
          fprintf(fp,"%lf %lf %lf %d ",x[3*i],x[3*i+1],x[3*i+2],iblank[i]);
          for (int j = 0; j < nvar; j++)
            fprintf(fp,"%lf ",q[j*nnodes+i]);
          fprintf(fp,"\n");
        }
    }
  ba=1-BASE;
  for (int n = 0; n < ntypes; n++)
    {
      nvert=nv[n];
      for (int i = 0; i < nc[n]; i++)
	{
	  if (nvert==4)
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+3]+ba);
	    }
	  else if (nvert==5) 
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+4]+ba);
	    }
	  else if (nvert==6) 
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+5]+ba,
		      vconn[n][nvert*i+5]+ba);
	    }
	  else if (nvert==8)
	    {
	      fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		      vconn[n][nvert*i]+ba,
		      vconn[n][nvert*i+1]+ba,
		      vconn[n][nvert*i+2]+ba,
		      vconn[n][nvert*i+3]+ba,
		      vconn[n][nvert*i+4]+ba,
		      vconn[n][nvert*i+5]+ba,
		      vconn[n][nvert*i+6]+ba,
		      vconn[n][nvert*i+7]+ba);
	    }
	}
    }
  fclose(fp);
  return;
}
  
void MeshBlock::getWallBounds(int *mtag,int *existWall, double wbox[6])
{
  *mtag = meshtag - BASE; // + 1?
  if (nwbc <= 0)
  {
    *existWall = 0;
    for (int i = 0; i < 3; i++)
    {
      wbox[i]   = BIGVALUE;
      wbox[3+i] = -BIGVALUE;
    }
    return;
  }

  *existWall=1;
  wbox[0] = wbox[1] = wbox[2] =  BIGVALUE;
  wbox[3] = wbox[4] = wbox[5] = -BIGVALUE;

  for (int i = 0; i < nwbc; i++)
  {
    int inode = wbcnode[i]-BASE;
    int i3 = 3*inode;
    for (int j = 0; j < 3; j++)
    {
      wbox[j] = min(wbox[j],x[i3+j]);
      wbox[j+3] = max(wbox[j+3],x[i3+j]);
    }
  }
}

void MeshBlock::getOversetBounds(int *mtag, int *existOver, double obox[6])
{
  *mtag = meshtag - BASE;
  if (nobc <= 0)
  {
    *existOver=0;
    for (int i = 0; i < 3; i++)
    {
      obox[i]   =  BIGVALUE;
      obox[3+i] = -BIGVALUE;
    }
    return;
  }

  *existOver = 1;
  obox[0] = obox[1] = obox[2] =  BIGVALUE;
  obox[3] = obox[4] = obox[5] = -BIGVALUE;

  for (int i = 0; i < nobc; i++)
  {
    int ind = 3*obcnode[i]-BASE;
    for (int j = 0; j < 3; j++)
    {
      obox[j]   = min(obox[j],   x[ind+j]);
      obox[j+3] = max(obox[j+3], x[ind+j]);
    }
  }
}
  
void MeshBlock::markWallBoundary(int *sam,int nx[3],double extents[6])
{
  std::vector<int> iflag(ncells);
  std::vector<int> inode(nnodes);

  // Mark all wall boundary nodes
  for (int i = 0; i < nwbc; i++)
  {
    int ii = wbcnode[i]-BASE;
    inode[ii] = 1;
  }

  // Mark all wall boundary cells (Any cell with wall-boundary node)
  int m = 0;
  for (int n = 0; n < ntypes; n++)
  {
    int nvert = nv[n];
    for (int i = 0; i < nc[n]; i++)
    {
      for (int j = 0; j < nvert; j++)
      {
        int ii = vconn[n][nvert*i+j]-BASE;
        if (inode[ii] == 1)
        {
          iflag[m] = 1;
          break;
        }
      }
      m++;
    }
  }

  // find delta's in each directions
  double ds[3];
  for (int k = 0; k < 3; k++) ds[k] = (extents[k+3]-extents[k])/nx[k];

  // mark sam cells with wall boundary cells now
  int imin[3];
  int imax[3];
  m = 0;
  for (int n = 0; n < ntypes; n++)
  {
    int nvert = nv[n];
    for (int i = 0; i < nc[n]; i++)
    {
      if (iflag[m] == 1)
      {
        // find the index bounds of each wall boundary cell bounding box
        imin[0] = imin[1] = imin[2] =  BIGINT;
        imax[0] = imax[1] = imax[2] = -BIGINT;
        for (int j = 0; j < nvert; j++)
        {
          int i3 = 3*(vconn[n][nvert*i+j]-BASE);
          for (int k = 0; k < 3; k++)
          {
            double xv = x[i3+k];
            int iv = floor((xv-extents[k])/ds[k]);
            imin[k] = min(imin[k],iv);
            imax[k] = max(imax[k],iv);
          }
        }

        for (int j = 0; j < 3; j++)
        {
          imin[j] = max(imin[j],0);
          imax[j] = min(imax[j],nx[j]-1);
        }

        // mark sam to 2
        for (int kk = imin[2]; kk < imax[2]+1; kk++)
        {
          for (int jj = imin[1]; jj < imax[1]+1; jj++)
          {
            for (int ii = imin[0]; ii < imax[0]+1; ii++)
            {
              int mm = (kk*nx[1] + jj)*nx[0] + ii;
              sam[mm] = 2;
            }
          }
        }
      }
      m++;
    }
  }
}

void MeshBlock::markOversetBoundary(int *sam,int nx[3],double extents[6])
{
  std::vector<int> iflag(ncells);
  std::vector<int> inode(nnodes);

  // Mark all wall boundary nodes
  for (int i = 0; i < nobc; i++) /// TESTING
  {
    int ii = obcnode[i]-BASE; /// TESTING
    inode[ii] = 1;
  }

  // Mark all wall boundary cells (Any cell with wall-boundary node)
  int m = 0;
  for (int n = 0; n < ntypes; n++)
  {
    int nvert = nv[n];
    for (int i = 0; i < nc[n]; i++)
    {
      for (int j = 0; j < nvert; j++)
      {
        int ii = vconn[n][nvert*i+j]-BASE;
        if (inode[ii] == 1)
        {
          iflag[m] = 1;
          break;
        }
      }
      m++;
    }
  }

  // find delta's in each directions
  double ds[3];
  for (int k = 0; k < 3; k++) ds[k] = (extents[k+3]-extents[k])/nx[k];

  // mark sam cells with wall boundary cells now
  int imin[3];
  int imax[3];
  m = 0;
  for (int n = 0; n < ntypes; n++)
  {
    int nvert = nv[n];
    for (int i = 0; i < nc[n]; i++)
    {
      if (iflag[m] == 1)
      {
        // find the index bounds of each wall boundary cell bounding box
        imin[0] = imin[1] = imin[2] =  BIGINT;
        imax[0] = imax[1] = imax[2] = -BIGINT;
        for (int j = 0; j < nvert; j++)
        {
          int i3 = 3*(vconn[n][nvert*i+j]-BASE);
          for (int k = 0; k < 3; k++)
          {
            double xv = x[i3+k];
            int iv = floor((xv-extents[k])/ds[k]);
            imin[k] = min(imin[k],iv);
            imax[k] = max(imax[k],iv);
          }
        }

        for (int j = 0; j < 3; j++)
        {
          imin[j] = max(imin[j],0);
          imax[j] = min(imax[j],nx[j]-1);
        }

        // mark sam to 2
        for (int kk = imin[2]; kk < imax[2]+1; kk++)
        {
          for (int jj = imin[1]; jj < imax[1]+1; jj++)
          {
            for (int ii = imin[0]; ii < imax[0]+1; ii++)
            {
              int mm = (kk*nx[1] + jj)*nx[0] + ii;
              sam[mm] = 2;
            }
          }
        }
      }
      m++;
    }
  }
}

void MeshBlock::getReducedOBB(OBB *obc,double *realData) 
{
  double bbox[6],xd[3];

  for (int j = 0; j < 3; j++)
  {
    realData[j]   = BIGVALUE;
    realData[j+3] = -BIGVALUE;
  }

  for (int n = 0; n < ntypes; n++)
  {
    int nvert = nv[n];
    for (int i = 0; i < nc[n]; i++)
    {
      bbox[0] = bbox[1] = bbox[2] =  BIGVALUE;
      bbox[3] = bbox[4] = bbox[5] = -BIGVALUE;

      // Get the bbox of the cell in *obc*'s coordinate system
      for (int m = 0; m < min(nvert,8); m++)
      {
        int i3 = 3*(vconn[n][nvert*i+m]-BASE);
        for (int j = 0; j < 3; j++)
          xd[j] = 0;

        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++)
            xd[j] += (x[i3+k]-obc->xc[k])*obc->vec[j][k];

        for (int j = 0; j < 3; j++)
        {
          bbox[j]   = min(bbox[j],xd[j]);
          bbox[j+3] = max(bbox[j+3],xd[j]);
        }
      }

      // Check for intersection with the OBB 'obc'
      int iflag = 0;
      for (int j = 0; j < 3; j++)
      {
        iflag = (iflag || (bbox[j] > obc->dxc[j]));
        iflag = (iflag || (bbox[j+3] < -obc->dxc[j]));
      }

      if (iflag) continue;

      // Get extents of bbox in *our* OBB axes
      for (int m = 0; m < min(nvert,8); m++)
      {
        int i3 = 3*(vconn[n][nvert*i+m]-BASE);
        for (int j = 0; j < 3; j++)
          xd[j] = 0;

        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++)
            xd[j] += (x[i3+k]-obb->xc[k])*obb->vec[j][k];

        for (int j = 0; j < 3; j++)
        {
          realData[j]   = min(realData[j],xd[j]);
          realData[j+3] = max(realData[j+3],xd[j]);
        }
      }
    }
  }

  // Get new xc, dxc of reduced OBB in our OBB's axes
  for (int j = 0; j < 6; j++)
    bbox[j] = realData[j];

  for (int j = 0; j < 3; j++)
  {
    realData[j] = obb->xc[j];

    for (int k = 0; k < 3; k++) // new centroid of reduced obb
      realData[j] += ((bbox[k]+bbox[k+3])*0.5)*obb->vec[k][j];

    realData[j+3] = (bbox[j+3]-bbox[j])*0.51;
  }
}

void MeshBlock::getReducedOBB_SuperMesh(OBB *obc,double *realData)
{
  /* --- Get corner points of obb --- */

  std::vector<point> bpts(8);

  const int xsgn[8][3] = {{-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1}, {-1,-1,1}, {1,-1,1}, {1,1,1}, {-1,1,1}};

  // Add in offsets from centroid
  for (int n = 0; n < 8; n++)
    for (int i = 0; i < 3; i++)
      bpts[n][i] = xsgn[n][i]*obb->dxc[i];

  /* --- Get corner points of obc [transform to OBB coordinates] --- */

  // y = R1^T * x1 + c1 - c2
  std::vector<point> cpts_0(8);
  for (int i = 0; i < 8; i++)
    cpts_0[i] = point(obc->xc) - point(obb->xc);

  for (int n = 0; n < 8; n++)
  {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        cpts_0[n][i] += obc->vec[j][i]*(xsgn[n][j]*obc->dxc[j]);
  }

  // x2 = R2 * y
  std::vector<point> cpts(8);
  for (int n = 0; n < 8; n++)
  {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
          cpts[n][i] += obb->vec[i][j] * cpts_0[n][j];
  }

  /* --- Use a SuperMesh to find the intersection --- */

  SuperMesh mesh(bpts,cpts,1,3);

  auto Points = mesh.getSuperMeshPoints();

  /* --- Find the final reduced bbox of the intersection region --- */

  double bbox[6];
  bbox[0] = bbox[1] = bbox[2] =  BIGVALUE;
  bbox[3] = bbox[4] = bbox[5] = -BIGVALUE;
  for (auto &pt : Points)
  {
    for (int d = 0; d < 3; d++)
    {
      bbox[d]   = min(bbox[d],   pt[d]);
      bbox[d+3] = max(bbox[d+3], pt[d]);
    }
  }

  // Get new xc, dxc of reduced OBB in our OBB's axes
  for (int i = 0; i < 3; i++)
  {
    realData[i] = obb->xc[i];

    for (int d = 0; d < 3; d++) // new centroid of reduced obb
      realData[i] += ((bbox[d]+bbox[d+3])*0.5)*obb->vec[d][i];

    realData[i+3] = (bbox[i+3]-bbox[i])*0.5; // Adding 5% to extents
  }
}

void MeshBlock::getQueryPoints(OBB *obc,
			       int *nints,int **intData,
			       int *nreals, double **realData)
{
  double xd[3];

  std::vector<int> inode(nnodes);

  *nints=*nreals=0; 

  for (int i = 0; i < nnodes; i++)
  {
    int i3 = 3*i;
    for (int j = 0; j < 3; j++) xd[j]=0;
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        xd[j]+=(x[i3+k]-obc->xc[k])*obc->vec[j][k];

    if (fabs(xd[0]) <= obc->dxc[0] &&
        fabs(xd[1]) <= obc->dxc[1] &&
        fabs(xd[2]) <= obc->dxc[2])
    {
      inode[*nints]=i;
      (*nints)++;
      (*nreals)+=3; //4;
    }
  }

  (*intData)=(int *)malloc(sizeof(int)*(*nints));
  (*realData)=(double *)malloc(sizeof(double)*(*nreals));

  int m = 0;
  for (int i = 0; i < *nints; i++)
  {
    int i3 = 3*inode[i];
    (*intData)[i]=inode[i];
    (*realData)[m++]=x[i3];
    (*realData)[m++]=x[i3+1];
    (*realData)[m++]=x[i3+2];
    //(*realData)[m++]=nodeRes[inode[i]];
  }
}  
  
void MeshBlock::writeOBB(int bid)
{
  FILE *fp;
  char intstring[7];
  char fname[80];
  int l,k,j,m,il,ik,ij;
  REAL xx[3];

  sprintf(intstring,"%d",100000+bid);
  sprintf(fname,"box%s.dat",&(intstring[1]));
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"Box file\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEPOINT\n",8,
	  1);

  for(l=0;l<2;l++)
    {
      il=2*(l%2)-1;
      for (int k = 0; k < 2; k++)
	{
	  ik=2*(k%2)-1;
	  for (int j = 0; j < 2; j++)
	    {
	      ij=2*(j%2)-1;
	      xx[0]=xx[1]=xx[2]=0;
	      for (int m = 0; m < 3; m++)
		xx[m]=obb->xc[m]+ij*obb->vec[0][m]*obb->dxc[0]
		  +ik*obb->vec[1][m]*obb->dxc[1]
		  +il*obb->vec[2][m]*obb->dxc[2];	      
	      fprintf(fp,"%f %f %f\n",xx[0],xx[1],xx[2]);
	    }
	}
    }
  fprintf(fp,"1 2 4 3 5 6 8 7\n");
  fprintf(fp,"%e %e %e\n",obb->xc[0],obb->xc[1],obb->xc[2]);
  for (int k = 0; k < 3; k++)
   fprintf(fp,"%e %e %e\n",obb->vec[0][k],obb->vec[1][k],obb->vec[2][k]);
  fprintf(fp,"%e %e %e\n",obb->dxc[0],obb->dxc[1],obb->dxc[2]);
  fclose(fp);
}

int MeshBlock::findPointDonor(double *x_pt)
{
  int foundCell;
  double rst[3] = {0.0};
  adt->searchADT(this,&foundCell,x_pt,rst);
  return foundCell;
}

std::unordered_set<int> MeshBlock::findCellDonors(double *bbox)
{
  std::unordered_set<int> foundCells;
  adt->searchADT_box(elementList.data(),foundCells,bbox);
  return foundCells;
}

//
// destructor that deallocates all the
// the dynamic objects inside
//
MeshBlock::~MeshBlock()
{
  //
  // free all data that is owned by this MeshBlock
  // i.e not the pointers of the external code.
  //
  if (adt) delete[] adt;
  if (donorList) {
    for (int i = 0; i < nnodes; i++) deallocateLinkList(donorList[i]);
    free(donorList);
  }
  return;  /// Why is this here??
  if (interpListCart) delete [] interpListCart;
  if (obb) free(obb);
  if (interp2donor) free(interp2donor);
  if (cancelList) deallocateLinkList2(cancelList);
  if (ctag) free(ctag);
  if (ftag) free(ftag);
  if (!iartbnd) free(iblank_cell);
  if (pointsPerCell) free(pointsPerCell);
  if (pointsPerFace) free(pointsPerFace);
  if (picked) free(picked);
  if (rxyzCart) free(rxyzCart);
  if (donorIdCart) free(donorIdCart);
  if (pickedCart) free(pickedCart);
  if (ctag_cart) free(ctag_cart);

  // need to add code here for other objects as and
  // when they become part of MeshBlock object  
}

//
// set user specified node and cell resolutions
//
void MeshBlock::setResolutions(double *nres,double *cres)
{
  userSpecifiedNodeRes=nres;
  userSpecifiedCellRes=cres;
}

void MeshBlock::setTransform(double* mat, double* pvt, double* off, int ndim)
{
  if (ndim != nDims)
    ThrowException("MeshBlock::set_transform: input ndim != nDims");
  
  // adding pivot point
  rrot = true;
  Rmat = mat;
  Pivot = pvt;
  offset = off;

  //printf("mb pvx %lf %lf %lf\n", pvt[0], pvt[1], pvt[2]);
  
  // adding pivot point
  if (adt)  {
    adt->setTransform(mat,pvt,off,ndim);
  }
}

void MeshBlock::set_soasz(unsigned int sz) {
  soasz = sz;
}

void MeshBlock::set_maxnface_maxnfpts(unsigned int nface_in, unsigned int nfpts_in) {
  maxnface = nface_in;
  maxnfpts = nfpts_in;
}

void MeshBlock::set_face_numbers(unsigned int nmpif, unsigned int nbcf) {
  nmpifaces = nmpif;
  nbcfaces = nbcf;
}

void MeshBlock::set_face_fpts(int* ffpts, unsigned int ntface) {
  face_fpts.resize(ntface);
  std::copy(ffpts, ffpts + ntface, face_fpts.begin());
}

void MeshBlock::set_fcelltypes(int* fctype, unsigned int ntface) {
  // this is a one time deal, no need to parallel
  fcelltypes = std::vector<std::vector<int>> (ntface, std::vector<int>(2,-1));
  for(int i=0;i<ntface;++i){
    fcelltypes[i][0] = fctype[i*2+0];
    fcelltypes[i][1] = fctype[i*2+1];
  }
}

void MeshBlock::set_fposition(int* fpos, unsigned int ntface) {
  fposition = std::vector<std::vector<int>> (ntface, std::vector<int>(2, -1));
  for(int i=0;i<ntface;++i) {
    fposition[i][0] = fpos[i*2+0];
    fposition[i][1] = fpos[i*2+1];
  }
}

void MeshBlock::set_facecoords_mapping(unsigned long long int basedata,
                                        int* faceinfo, int* mapping, int nfpts
                                       ) {
  fcoords_basedata = basedata;
  int etype, cidx, fpos, nid;
  for(int i=0;i<nfpts;++i) {
    etype = faceinfo[i*4+0]; // element type
    cidx = faceinfo[i*4+1];  // element number in all cell types
    fpos = faceinfo[i*4+2];  // face position in the cell
    nid = faceinfo[i*4+3];   // node idx

    auto key = std::vector<int>{etype, cidx, fpos, nid};
    auto it = fcoords_mapping.find(key);
    if(it == fcoords_mapping.end()) {
      fcoords_mapping.insert({key, mapping[i]});
    }
  }
}



void MeshBlock::set_interior_mapping(unsigned long long int basedata, 
                                     unsigned long long int grad_basedata,
                                     int* faceinfo, 
                                     int* mapping,
                                     int* grad_mapping,
                                     int* grad_strides,
                                     int nfpts) {
  interior_basedata = basedata;
  interior_grad_basedata = grad_basedata;

  int etype, cidx, fpos, nid;
  for(int i=0;i<nfpts;++i) {
    etype = faceinfo[i*4+0]; // element type
    cidx = faceinfo[i*4+1];  // element number in all cell types
    fpos = faceinfo[i*4+2];  // face position in the cell
    nid = faceinfo[i*4+3];   // node idx

    auto key = std::vector<int>{etype, cidx, fpos, nid};
    {
      auto it = interior_mapping.find(key);
      if(it == interior_mapping.end()) {
        interior_mapping.insert({key, mapping[i]});
      }
    }
    
    {
      auto it = interior_grad_mapping.find(key);
      if(it == interior_grad_mapping.end()) {
        interior_grad_mapping.insert({key, grad_mapping[i]});
      }
    }

    {
      auto it = interior_grad_strides.find(key);
      if(it == interior_grad_strides.end()) {
        interior_grad_strides.insert({key, grad_mapping[i]});
      }
    }
  }
}

void MeshBlock::figure_out_interior_artbnd_target() {
  int nfringe = interior_ab_faces.size();
  if (nfringe == 0) return;
  // interior_mapping is for every flux points
  auto range = std::views::iota(0, nfringe);
  interior_target_nfpts.resize(nfringe);
  interior_target_scan.resize(nfringe);

  // we need face_fpts
  std::for_each(std::execution::par, range.begin(), range.end(),
    [this] (auto idx) {
      auto fid = interior_ab_faces[idx].first;
      interior_target_nfpts[idx] = face_fpts[fid];
    });   
  
  //
  std::exclusive_scan(std::execution::par, 
      interior_target_nfpts.begin(), interior_target_nfpts.end(), 
      interior_target_scan.begin(), 0
    );

  // total number of fpts
  auto tnfpts = interior_target_scan[nfringe-1] + interior_target_nfpts[nfringe-1];
  interior_target_mapping.resize(tnfpts);
  interior_target_grad_mapping.resize(tnfpts);
  interior_target_grad_strides.resize(tnfpts);
  interior_tnfpts = tnfpts;

  int rank =0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // for these interior fringe faces
  // here we need fcelltypes and fposition
  std::for_each(std::execution::par, range.begin(), range.end(),
    [this, &rank](auto idx) {
      auto fid = interior_ab_faces[idx].first;
      auto c0 = f2c[fid*2+0];
      auto c1 = f2c[fid*2+1];
      auto ib0 = iblank_cell[c0];
      auto ib1 = iblank_cell[c1];
        
      auto check = ib0 ^ ib1;
      if(check == 0) printf(" something is wrong on blanking status\n");         

      if(ib0 == 0) {
        for(auto i=0;i<face_fpts[fid];++i) {
          auto key = std::vector<int>{fcelltypes[fid][0], c0, fposition[fid][0], i};
          auto npid = interior_target_scan[idx] + i;
          {
            auto it = interior_mapping.find(key);
            interior_target_mapping[npid] = it->second;
          }
          {
            auto it = interior_grad_mapping.find(key);
            interior_target_grad_mapping[npid] = it->second;
          }
          {
            auto it = interior_grad_strides.find(key);
            interior_target_grad_strides[npid] = it->second;
          }
        }
      } else {
        for(auto i=0;i<face_fpts[fid];++i) {
          auto key = std::vector<int>{fcelltypes[fid][1], c1, fposition[fid][1], i};
          auto npid = interior_target_scan[idx] + i;
          {
            auto it = interior_mapping.find(key);
            interior_target_mapping[npid] = it->second;
          }
          {
            auto it = interior_grad_mapping.find(key);
            interior_target_grad_mapping[npid] = it->second;
          }
          {
            auto it = interior_grad_strides.find(key);
            interior_target_grad_strides[npid] = it->second;
          }
        }
      }
    });

    // now we copy this to device 
    interior_target_mapping_d.resize(interior_target_mapping.size());
    interior_target_mapping_d.assign(interior_target_mapping.data(), interior_target_mapping.size(), NULL);

    interior_target_grad_mapping_d.resize(interior_target_grad_mapping.size());
    interior_target_grad_mapping_d.assign(interior_target_grad_mapping.data(), interior_target_grad_mapping.size(), NULL);

    interior_target_grad_strides_d.resize(interior_target_grad_strides.size());
    interior_target_grad_strides_d.assign(interior_target_grad_strides.data(), interior_target_grad_strides.size(), NULL);

}

void MeshBlock::set_mpi_mapping(unsigned long long int basedata,  int* faceinfo, int* mapping, int nfpts) {
  //one pyfr partition could have different mpi inters, each mpi inters 
  //have its own memory allocated such that there are not necessarily continuous
  //moreover, the mpi variable are stored variable by variable 
  //Therefore here int mapping is organized as mapping[i*nv + 0]-> mapping[i*nv+nv]
  //for the different variables at the same flux point
  //Note that mapping here is interms of char instead of float/double in case the 
  //discontinuous memory is not float/double aligned
  mpi_basedata = basedata;
  int etype, cidx, fpos, nid;
  for(int i=0;i<nfpts;++i) {
    etype = faceinfo[i*4+0]; // element type
    cidx = faceinfo[i*4+1];  // element number
    fpos = faceinfo[i*4+2];  // face position
    nid = faceinfo[i*4+3];   // node id
    
    auto key = std::vector<int>{etype, cidx, fpos, nid};
    auto it = mpi_mapping.find(key);
    if(it == mpi_mapping.end()) {
      mpi_mapping.insert({key, mapping[i]});
    }
  }
  mpi_entire_tnfpts = nfpts;
  mpi_entire_mapping_h = std::vector<int>(mapping, mapping + nfpts);
  mpi_entire_mapping_d.assign(mpi_entire_mapping_h.data(), mpi_entire_mapping_h.size(), NULL);
}

void MeshBlock::set_mpi_rhs_mapping(unsigned long long int basedata,int* mapping,int* strides, int nfpts) {
  if(nfpts != mpi_entire_tnfpts) {
    printf("something is wrong with missmatching nfpts for mpi rhs nfpts %d mpi_entire_tnfpts %d\n", nfpts, mpi_entire_tnfpts);
    int pid = getpid();
    printf("current pid is %d hang at inconsistent mpi\n",pid);
    int idebugger = 0;
    while(idebugger) {

    };
  }
  mpi_rhs_basedata = basedata;
  mpi_entire_rhs_mapping.resize(nfpts);
  mpi_entire_rhs_strides.resize(nfpts);
  mpi_entire_rhs_mapping_d.resize(nfpts);
  mpi_entire_rhs_strides_d.resize(nfpts);

  std::copy(mapping, mapping + nfpts, mpi_entire_rhs_mapping.data());
  std::copy(strides, strides + nfpts, mpi_entire_rhs_strides.data()); 
  
  mpi_entire_rhs_mapping_d.assign(mpi_entire_rhs_mapping.data(), mpi_entire_rhs_mapping.size(), NULL);
  mpi_entire_rhs_strides_d.assign(mpi_entire_rhs_strides.data(), mpi_entire_rhs_strides.size(), NULL);

}

void MeshBlock::set_overset_rhs_basedata(unsigned long long int basedata) {
   overset_rhs_basedata = basedata;
}

void MeshBlock::set_overset_mapping(unsigned long long int basedata,  int* faceinfo, int* mapping, int nfpts) {
  overset_basedata = basedata;
  int etype, cidx, fpos, nid;
  for(int i=0;i<nfpts;++i) {
    etype = faceinfo[i*4+0]; // element type
    cidx = faceinfo[i*4+1];  // element number
    fpos = faceinfo[i*4+2];  // face position
    nid = faceinfo[i*4+3];   // node id
    
    auto key = std::vector<int>{etype, cidx, fpos, nid};
    auto it = overset_mapping.find(key);
    if(it == overset_mapping.end()) {
      overset_mapping.insert({key, mapping[i]});
    }
  }
}
void MeshBlock::figure_out_facecoords_target() {
    // directly use nreceptorFaces and ftag to figure things out
  int nfringe = nreceptorFaces;
  if(nfringe == 0) return;
  auto range = std::views::iota(0, nfringe);
  fcoords_target_nfpts.resize(nfringe);
  fcoords_target_scan.resize(nfringe);

  std::for_each(std::execution::par, range.begin(), range.end(),
    [this] (auto idx) {
      auto fid = ftag[idx];
      fcoords_target_nfpts[idx] = face_fpts[fid];
    });   
  
  //
  std::exclusive_scan(std::execution::par, 
      fcoords_target_nfpts.begin(), fcoords_target_nfpts.end(), 
      fcoords_target_scan.begin(), 0
    );

  auto tnfpts = fcoords_target_scan[nfringe-1] + fcoords_target_nfpts[nfringe-1];
  fcoords_target_mapping.resize(tnfpts);
  fcoords_tnfpts = tnfpts;

  std::for_each(std::execution::par, range.begin(), range.end(),
    [this](auto idx) {
      auto fid = ftag[idx];
      auto c0 = f2c[fid*2+0];
        
      for(auto i=0;i<face_fpts[fid];++i) {
        auto key = std::vector<int>{fcelltypes[fid][0], c0, fposition[fid][0], i};
        auto it = fcoords_mapping.find(key);
        auto npid = fcoords_target_scan[idx] + i;
        fcoords_target_mapping[npid] = it->second;
      }
    });
  fcoords_target_mapping_d.resize(fcoords_target_mapping.size());
  fcoords_target_mapping_d.assign(fcoords_target_mapping.data(), fcoords_target_mapping.size(), NULL);
}

void MeshBlock::figure_out_mpi_artbnd_target() {
  int nfringe = mpi_ab_faces.size();
  if (nfringe == 0) return;
  auto range = std::views::iota(0, nfringe);
  mpi_target_nfpts.resize(nfringe);
  mpi_target_scan.resize(nfringe);
    
  std::for_each(std::execution::par, range.begin(), range.end(),
    [this] (auto idx) {
      auto fid = mpi_ab_faces[idx].first;
      mpi_target_nfpts[idx] = face_fpts[fid];
    });

  std::exclusive_scan(std::execution::par, 
      mpi_target_nfpts.begin(), mpi_target_nfpts.end(), 
      mpi_target_scan.begin(), 0
    );

  auto tnfpts = mpi_target_scan[nfringe-1] + mpi_target_nfpts[nfringe-1];
  mpi_target_mapping.resize(tnfpts);
  mpi_tnfpts = tnfpts;// this one will be used later
    
  int rank =0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::for_each(std::execution::par, range.begin(), range.end(),
    [this, &rank](auto idx) {
      auto fid = mpi_ab_faces[idx].first;
      auto c0 = f2c[fid*2+0];
      auto c1 = f2c[fid*2+1];
      auto ib0 = iblank_cell[c0];
        
      for(auto i=0;i<face_fpts[fid];++i) {
        auto key = std::vector<int>{fcelltypes[fid][0], c0, fposition[fid][0], i};
        auto it = mpi_mapping.find(key);
        auto npid = mpi_target_scan[idx] + i;
        mpi_target_mapping[npid] = it->second;
      }
    });
    
  mpi_target_mapping_d.resize(mpi_target_mapping.size());
  mpi_target_mapping_d.assign(mpi_target_mapping.data(), mpi_target_mapping.size(), NULL);
  
  // starting from here we figure out the indices of current mpi fpts
  mpi_target_rhs_fptsid.resize(mpi_tnfpts);
  std::for_each(std::execution::par, range.begin(), range.end(),
    [this, &rank] (auto idx) {
        auto fid = mpi_ab_faces[idx].first;
        for(auto i=0;i<face_fpts[fid];++i) {
            auto npid = mpi_target_scan[idx] + i;
            mpi_target_rhs_fptsid[npid] = npid; // get the id for current fpts
        }
    });
  mpi_target_rhs_fptsid_d.resize(mpi_tnfpts);
  mpi_target_rhs_fptsid_d.assign(mpi_target_rhs_fptsid.data(), mpi_target_rhs_fptsid.size(), NULL);
}


void MeshBlock::figure_out_overset_artbnd_target() {
  int nfringe = overset_ab_faces.size();
  if (nfringe == 0) return;
  auto range = std::views::iota(0, nfringe);
  overset_target_nfpts.resize(nfringe);
  overset_target_scan.resize(nfringe);
    
  std::for_each(std::execution::par, range.begin(), range.end(),
    [this] (auto idx) {
      auto fid = overset_ab_faces[idx].first;
      overset_target_nfpts[idx] = face_fpts[fid];
    });

  std::exclusive_scan(std::execution::par, 
      overset_target_nfpts.begin(), overset_target_nfpts.end(), 
      overset_target_scan.begin(), 0
    );

  auto tnfpts = overset_target_scan[nfringe-1] + overset_target_nfpts[nfringe-1];
  overset_target_mapping.resize(tnfpts);
 
  overset_tnfpts = tnfpts;// this one will be used later
    
  int rank =0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::for_each(std::execution::par, range.begin(), range.end(),
    [this, &rank](auto idx) {
      auto fid = overset_ab_faces[idx].first;
      auto c0 = f2c[fid*2+0];
      auto c1 = f2c[fid*2+1];
      auto ib0 = iblank_cell[c0];
        
      for(auto i=0;i<face_fpts[fid];++i) {
        auto key = std::vector<int>{fcelltypes[fid][0], c0, fposition[fid][0], i};
        auto it = overset_mapping.find(key);
        auto npid = overset_target_scan[idx] + i;
        overset_target_mapping[npid] = it->second;
      }
    });
    
    overset_target_mapping_d.resize(overset_target_mapping.size());
    overset_target_mapping_d.assign(overset_target_mapping.data(), overset_target_mapping.size(), NULL);
}

void MeshBlock::set_data_reorder_map(int* srted, int* unsrted, int ncells) {
  {
    int pid = getpid();
    printf("current pid is %d\n",pid);
    int idebugger = 0;
    while(idebugger) {

    };
  }

  srted_order = std::vector<std::vector<std::vector<int>>> (ncells, 
    std::vector<std::vector<int>>(maxnface, std::vector<int>(maxnfpts,-1))
  );

  unsrted_order = std::vector<std::vector<std::vector<int>>> (ncells, 
    std::vector<std::vector<int>>(maxnface, std::vector<int>(maxnfpts,-1))
  );
  
  for(auto i=0;i<ncells;++i) {
    for(auto j=0;j<maxnface;++j) {
      for(auto k=0;k<maxnfpts;++k){
        srted_order[i][j][k] = srted[i*maxnface*maxnfpts+j*maxnfpts+k];
      } 
    }
  }

  for(auto i=0;i<ncells;++i) {
    for(auto j=0;j<maxnface;++j) {
      for(auto k=0;k<maxnfpts;++k){
        unsrted_order[i][j][k] = unsrted[i*maxnface*maxnfpts+j*maxnfpts+k];
      } 
    }
  }

  // we know in unsorted_order the id of fpts always in acending order
  // data being passed in is in unsrted order  
  // the index of a point in srted in unsrted
  std::vector<std::vector<std::unordered_map<int, int>>> unsrted_to_srted_map(
        ncells, std::vector<std::unordered_map<int, int>> (maxnface)
    );
  auto range = std::views::iota(0, ncells);
  // ntypes // number of different types
  // nc // number of cells per type
  // ncf // number of faces per type
  std::for_each(std::execution::par, range.begin(), range.end(), 
    [this, &unsrted_to_srted_map](auto idx) {
      for(auto i=0;i<ncf[0];++i) { // use maxnface here for now
        std::unordered_map<int, int> lmap;
        auto fid = c2f[0][idx*maxnface + i]; // current face id
        auto nfpts = face_fpts[fid];
        for(auto j=0;j<nfpts;++j) {
          auto sid = srted_order[idx][i][j]; // the sorted id of flux points
          // search this id in undorted_order[idx][i]
          auto& pool = unsrted_order[idx][i];
          auto it = std::lower_bound(pool.begin(), pool.begin() + nfpts, sid);
          auto id = std::distance(pool.begin(), it);
          if(id >= nfpts) printf("something is wrong on finding the reorder key\n");
          lmap[sid] = id;
        }
        unsrted_to_srted_map[idx][i] = lmap;
      }
    });
   
  face_unsrted_to_srted_map = unsrted_to_srted_map;
 
}

void MeshBlock::reset_entire_mpi_face_artbnd_status_pointwise(unsigned int nvar) {
  if(nmpifaces == 0) return;
  double* status = reinterpret_cast<double*>(mpi_basedata);
  int* mapping = mpi_entire_mapping_d.data();
  reset_mpi_face_artbnd_status_wrapper(status,mapping,1.0,0,mpi_entire_tnfpts,nvar,soasz,3);
}

void MeshBlock::reset_mpi_face_artbnd_status_pointwise(unsigned int nvar) {
  if(mpi_tnfpts == 0) return;
  double* status = reinterpret_cast<double*>(mpi_basedata);
  int* mapping = mpi_target_mapping_d.data();
  reset_mpi_face_artbnd_status_wrapper(status, mapping, 1.0, 0,  mpi_tnfpts, nvar, soasz, 3);
}

void MeshBlock::prepare_mpi_artbnd_target_data(double* data, int nvar) {
  // here we need to do some reordering of the data
  // we know for now the mpi faces are in the first mpi_tnfpts for cases we support
  int nfringe = mpi_ab_faces.size();
  if(nfringe == 0) return;
  mpi_data_h.resize(mpi_tnfpts*nvar);
  auto range = std::views::iota(0, nfringe);  
  
  std::for_each(std::execution::par, range.begin(), range.end(), [this, nvar, data](auto idx) {
    auto  sidx = mpi_target_scan[idx];    // start idx of each face
    auto  nfpt = mpi_target_nfpts[idx];
    auto  fid = mpi_ab_faces[idx].first; // global id
    auto  c0 = f2c[fid*2+0];
    auto  fp = fposition[fid][0]; // face position
    auto& srted = srted_order[c0][fp];  // srted ids for current face
    // now swap the values
    for(auto i=0;i<nfpt;++i) {
        auto srtid = srted[i];
        //auto unsrtid = lmap[srtid];
        auto unsrtid = face_unsrted_to_srted_map[c0][fp][srtid];
        for(auto k=0;k<nvar;++k) {
            mpi_data_h[sidx*nvar + i*nvar + k] = data[sidx*nvar + unsrtid * nvar + k];
        }
    }
  });
  // then copy this data to the device
  mpi_data_d.resize(mpi_data_h.size());
  mpi_data_d.assign(mpi_data_h.data(), mpi_data_h.size(), NULL);

  //for(auto i=0;i<mpi_tnfpts*nvar;++i) printf("original data %12.8e\n", data[i]);
  //for(auto i: mpi_data_h) printf("data %12.8e\n", i);
    
  // now copy the data to the destination we want
  double* dst = reinterpret_cast<double*>(mpi_rhs_basedata);
  double* src = mpi_data_d.data(); 
 
  int* mapping = mpi_entire_rhs_mapping_d.data();
  int* strides = mpi_entire_rhs_strides_d.data();
  
  int* fptsids = mpi_target_rhs_fptsid_d.data();
  pointwise_copy_to_mpi_rhs_wrapper(dst, mapping, strides, src, fptsids, mpi_tnfpts, nvar, 3);
}

void MeshBlock::prepare_overset_artbnd_target_data(double* data, int nvar) {
  // currently, we assume the overset grid will not be cut by any other grids
  // hence the data are all for overset boundaries
  int nfringe = overset_ab_faces.size();
  if(nfringe == 0 ) return;
  overset_data_h.resize(overset_tnfpts*nvar);
  auto range = std::views::iota(0, nfringe);  

  std::for_each(std::execution::par, range.begin(), range.end(), [this, nvar, data](auto idx) {
    auto  sidx = overset_target_scan[idx];    // start idx of each face
    auto  nfpt = overset_target_nfpts[idx];
    auto  fid = overset_ab_faces[idx].first; // global id
    auto  c0 = f2c[fid*2+0];
    auto  fp = fposition[fid][0]; // face position
    auto& srted = srted_order[c0][fp];  // srted ids for current face
    // now swap the values
    for(auto i=0;i<nfpt;++i) {
        auto srtid = srted[i];
        auto unsrtid = face_unsrted_to_srted_map[c0][fp][srtid];
        for(auto k=0;k<nvar;++k) {
            overset_data_h[k*overset_tnfpts+sidx+i] = data[sidx*nvar + unsrtid * nvar + k];
        }
    }
  });
  // then copy this data to the device
  // overset_data_d.resize(overset_data_h.size());
  // overset_data_d.assign(overset_data_h.data(), overset_data_h.size(), NULL);

  // overset data is directly copied to the desination
  // we are assuming the overset grid is not cut by any other grid for now
  double* dst = reinterpret_cast<double*>(overset_rhs_basedata);
  cuda_copy_h2d(dst, overset_data_h.data(), overset_data_h.size());
}


void MeshBlock::prepare_interior_artbnd_target_data(double* data, int nvar) {
  int idebugger = 0;
  int pid = getpid();
  while(idebugger) {

  }
  // now, we want to prepare the data  for interior artbnd
  // data, mpi_tnfpts * nvar are the data for mpi fringe faces
  if(interior_tnfpts == 0) return;
  //interior_data_d.resize(interior_tnfpts * nvar); 
  interior_data_d.assign(data + mpi_tnfpts * nvar, interior_tnfpts * nvar, NULL);
  unpack_interior_artbnd_u_pointwise(nvar);
}

void MeshBlock::prepare_interior_artbnd_target_data_gradient(double* data, int nvar, int dim) {
  if(interior_tnfpts == 0) return;
  interior_data_grad_d.resize(interior_tnfpts * nvar * dim);
  interior_data_grad_d.assign(data + mpi_tnfpts*nvar*dim, interior_tnfpts*nvar*dim, NULL);
  unpack_interior_artbnd_du_pointwise(nvar, dim);
}

void MeshBlock::unpack_interior_artbnd_u_pointwise(unsigned int nvar) {
  if(interior_tnfpts == 0) return;
  double* usrc = interior_data_d.data();
  double* udst = reinterpret_cast<double*>(interior_basedata);
  int* mapping = interior_target_mapping_d.data();
  unpack_fringe_u_wrapper(usrc, udst, mapping, 0, interior_tnfpts, nvar, soasz, 3); 
}

void MeshBlock::unpack_interior_artbnd_du_pointwise(unsigned int nvar, unsigned int dim) {
  if(interior_tnfpts == 0) return;
  double* usrc = interior_data_grad_d.data();
  double* udst = reinterpret_cast<double*>(interior_grad_basedata);
  int* mapping = interior_target_grad_mapping_d.data();
  int* strides = interior_target_grad_strides_d.data();
  unpack_fringe_grad_wrapper(usrc, udst, mapping, strides, 0, interior_tnfpts, nvar, dim, soasz, 3);
}

void MeshBlock::pack_fringe_facecoords_pointwise(double* rxyz) {

  if(nreceptorFaces != 0) {
    figure_out_facecoords_target();
  } else {
    fcoords_tnfpts = 0;
  }

  if(fcoords_tnfpts == 0) return;
  // resize the work data first
  constexpr int dim = 3;
  fcoords_data_d.resize(fcoords_tnfpts * dim);
  double* src = reinterpret_cast<double*>(fcoords_basedata);
  double* dst = fcoords_data_d.data();
  int* mapping = fcoords_target_mapping_d.data();
  pack_fringe_coords_wrapper(mapping, dst, src, fcoords_tnfpts, dim, soasz, 3);
  // now also copy the data to host
  cuda_copy_d2h(dst, rxyz, fcoords_tnfpts*dim);
}

