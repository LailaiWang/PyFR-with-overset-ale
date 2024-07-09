# ifndef MESHBLOCK_H
#define MESHBLOCK_H
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
/**
 * MeshBlock class - container and functions for generic unstructured grid partition in 3D
 *         
 * Jay Sitaraman
 * 02/20/2014
 */
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <vector>
#include <mutex>

#include <algorithm>
#include <ranges>
#include <execution>

#include "codetypes.h"
#include "funcs.hpp"
#include "points.hpp"
#include "ADT.h"

#ifdef _GPU
#include "cuda_funcs.h"
#include "dADT.h"
#include "dMeshBlock.h"
#endif

extern void reset_mpi_face_artbnd_status_wrapper(double*, int*, double, unsigned int, unsigned int, unsigned int, unsigned int, int);
extern void unpack_fringe_u_wrapper(double*, double*, int*, unsigned int, unsigned int, unsigned int, unsigned int, int);
extern void unpack_fringe_grad_wrapper(double*, double*, int*, int*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int);
extern void pointwise_copy_to_mpi_rhs_wrapper(double*, int*, int*, double*, int*, unsigned int, unsigned int, int);
extern void pack_fringe_coords_wrapper(int*, double*, double*, int, int, unsigned int ,int);

extern void pack_cell_coords_wrapper(int*, int*, double*, double*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, int);
extern void unpack_unblank_u_wrapper(int*, int*, double*, double*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned itn, int);

struct vector_hash {
  int operator()(const std::vector<int> &V) const {
    int hash = V.size();
    for(auto &i : V) {
      hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

//! Helper struct for direct-cut method [Galbraith 2013]
typedef struct CutMap
{
  int type;                  //! Cut type: Solid wall (1) or overset bound (0)
  std::vector<int> flag;     //! Cut flag for all cells (essentially iblank)
  std::map<int,double> dist; //! Minimum distance to a cutting face
  std::map<int,int> nMin;    //! # of cut faces that are approx. 'dist' away
  std::map<int,Vec3> norm;   //! Normal vector of cutting face (or avg. of several)
  std::map<int,double> dot;   //! Dot prodcut of Normal vector with separation vector
} CutMap;

// forward declare to instantiate one of the methods
class parallelComm;
class CartGrid;

class MeshBlock
{
friend class dMeshBlock;
private:
  unsigned int soasz = 0;

  int nbcfaces;  /**< number of boundary faces*/
  int nmpifaces; /**< number of mpi faces*/

  int nnodes;  /** < number of grid nodes */
  int ncells;  /** < total number of cells */
  int nfaces;  /** < total number of faces (Art. Bnd.) */
  int ntypes;  /** < number of different types of cells */
  int nftype;  /** < number of different face types (triangle or quad) */
  int *nv;     /** < number of vertices for each type of cell */
  int *nc;     /** < number of each of different kinds of cells (tets, prism, pyramids, hex etc) */
  int *ncf;     /** < number of faces for each cell type */
  int *nf;     /** < number of faces of each type in grid*/
  int *nfv;    /** < number of vertices per face for each face type (3 or 4) */
  int nobc;    /** < number of overset boundary nodes */
  int nwbc;    /** < number of wall boundary nodes */
  //
  double *x;        /** < grid nodes x[3*nnodes] */

  //! Moving artbnd grid buffer variables
  std::vector<double> x2;
  std::vector<int> ibc_2, ibc_0;
  double *xtmp;
  int *ibc_tmp;
  double *vg;
  std::set<int> unblanks;

  int *iblank;      /** < iblank value for each grid node */
  int *iblank_cell; /** < iblank value at each grid cell */
  int *iblank_face; /** < iblank value at each grid face (Art. Bnd.) */
  //
  int **vconn;      /** < connectivity (cell to nodes) for each cell type */
  int **fconn;      /** < Connectivity (face to nodes) for each face type */
  int *f2c;         /** < Face to cell connectivity */
  int **c2f;         /** < Cell to face connectivity */
  int *wbcnode;     /** < wall boundary node indices */
  int *obcnode;     /** < overset boundary node indices */
  //
  std::vector<double> nodeRes;  /** < node resolution  */
  std::vector<double> obcRes;  /** < ref. mesh length at overset bndry nodes */
  double *userSpecifiedNodeRes;
  double *userSpecifiedCellRes;
  std::vector<double> elementBbox; /** < bounding box of the elements */
  std::vector<int> elementList;    /** < list of elements in */
  int ncells_adt;  //! Number of cells within ADT

  int nOverFaces;
  int nWallFaces;
  int nMpiFaces;
  int *overFaces; /** < List of explicitly-defined (by solver) overset faces */
  int *wallFaces; /** < List of explicitly-defined (by solver) solid wall faces */
  int *mpiFaces;  /** < List of MPI face IDs on rank */
  int *mpiFidR;   /** < Matching MPI face IDs on opposite rank */
  int *mpiProcR;  /** < Opposite rank for MPI face */

  std::vector<double> igbpdata; /** < List of x,y,z,ds for inter-grid bndry pts */

  std::vector<std::vector<int>> c2c;

  int nsend, nrecv;
  std::vector<int> sndMap, rcvMap;

  bool rrot = false;
  double* Rmat = NULL;
  double* Pivot = NULL;
  double* offset = NULL;

#ifdef _GPU
  dADT adt_d;       /** GPU-based ADT */
  dMeshBlock mb_d;  /** GPU-based mesh data */
#endif

  int maxnface; // max number of faces per element
  int maxnfpts; // max number of fpts per face

  // pyfr related
  std::vector<int> face_fpts; // number of fpts per face
  std::vector<std::vector<int>> fcelltypes; // left and right cell type
  std::vector<std::vector<int>> fposition;  // left and right face position

  std::unordered_map<std::vector<int>, int, vector_hash> interior_mapping;
  std::unordered_map<std::vector<int>, int, vector_hash> interior_grad_mapping;
  std::unordered_map<std::vector<int>, int, vector_hash> interior_grad_strides;

  std::unordered_map<std::vector<int>, int, vector_hash> mpi_mapping;
  std::unordered_map<std::vector<int>, int, vector_hash> overset_mapping;
  std::unordered_map<std::vector<int>, int, vector_hash> fcoords_mapping;

  std::vector<std::pair<int,int>> interior_ab_faces; //pair first->global id second loc id
  std::vector<std::pair<int,int>> mpi_ab_faces;
  std::vector<std::pair<int,int>> overset_ab_faces;
  
  unsigned long long int fcoords_basedata;
  int fcoords_tnfpts{0};
  std::vector<int> fcoords_target_nfpts;
  std::vector<int> fcoords_target_scan;
  std::vector<int> fcoords_target_mapping;
  dvec<int> fcoords_target_mapping_d;
  dvec<double> fcoords_data_d;

  unsigned long long int interior_basedata;
  unsigned long long int interior_grad_basedata;
  int interior_tnfpts{0};
  std::vector<int> interior_target_nfpts; // fringe nfpts per face
  std::vector<int> interior_target_scan; // staring idx of fpts on each face
  std::vector<int> interior_target_mapping; // mapping of very fpts 
  
  std::vector<int> interior_target_grad_mapping;
  std::vector<int> interior_target_grad_strides;

  dvec<int> interior_target_mapping_d; // interior mapping
  dvec<int> interior_target_grad_mapping_d;
  dvec<int> interior_target_grad_strides_d;
  dvec<double> interior_data_d; // memory buffer to interior ab
  dvec<double> interior_data_grad_d;

  unsigned long long int mpi_basedata; // address
  unsigned long long int mpi_rhs_basedata;
  int mpi_tnfpts{0}; // total number of mpi nfpts
  int mpi_entire_tnfpts{0};
  std::vector<int> mpi_target_nfpts;
  std::vector<int> mpi_target_scan;
  std::vector<int> mpi_target_mapping;
  dvec<int> mpi_target_mapping_d; 
  std::vector<int> mpi_entire_mapping_h;
  dvec<int> mpi_entire_mapping_d;
  std::vector<double> mpi_data_h;
  dvec<double> mpi_data_d; // memory buffer to interior ab
  
  std::vector<int> mpi_entire_nfpts;
  std::vector<int> mpi_entire_scan;

  std::vector<int> mpi_entire_rhs_mapping;
  std::vector<int> mpi_entire_rhs_strides;
  std::vector<int> mpi_target_rhs_fptsid;
  dvec<int> mpi_entire_rhs_mapping_d;
  dvec<int> mpi_entire_rhs_strides_d;
  dvec<int> mpi_target_rhs_fptsid_d;

  unsigned long long int overset_basedata;
  unsigned long long int overset_rhs_basedata;
  int overset_tnfpts{0};
  std::vector<int> overset_target_nfpts;
  std::vector<int> overset_target_scan;
  std::vector<int> overset_target_mapping;
  dvec<int> overset_target_mapping_d;
  std::vector<double> overset_data_h;
  dvec<double> overset_data_d;

  std::vector<std::vector<std::vector<int>>> srted_order;
  std::vector<std::vector<std::vector<int>>> unsrted_order;
  std::vector<std::vector<std::unordered_map<int, int>>> face_unsrted_to_srted_map;
    
  std::vector<int> celltypes; // 8 -> hex
  std::unordered_map<int, int> cell_nupts_per_type;
  std::unordered_map<int, std::vector<int>> cell_u_strides_per_type;
  std::unordered_map<int, std::vector<int>> cell_du_strides_per_type;
  std::unordered_map<int, unsigned long long int> cell_du_basedata_per_type;
  std::unordered_map<int, std::vector<int>> cell_coords_strides_per_type;
  std::unordered_map<int, unsigned long long int> cell_coords_basedata_per_type;
  std::vector<int> cell_target_coords_scan;
  dvec<int> cell_target_coords_scan_d;
  dvec<int> cell_target_ids_d;
  dvec<double> cell_target_coords_data_d;
  dvec<double> cell_target_soln_data_d;
  //
  // Alternating digital tree library
  //
  ADT *adt;   /** < Digital tree for searching this block */
  //
  DONORLIST **donorList;      /**< list of donors for the nodes of this mesh */
  //
  int ninterp;              /**< number of interpolations to be performed */
  int interpListSize;
  std::vector<INTERPLIST> interpList;   /**< list of donor nodes in my grid, with fractions and information of
                                 who they donate to */ 
  int *interp2donor;

  INTEGERLIST *cancelList;  /** receptors that need to be cancelled because of */
  int ncancel;              /** conflicts with the state of their donors */

   /* ---- Callback functions for high-order overset connectivity ---- */

  /*!
   * \brief Get the number of solution points in given cell
   */
  void (*get_nodes_per_cell)(int* cellID, int* nNodes);

  /*!
   * \brief Get the number of flux points on given face
   *        For the artificial boundary method
   */
  void (*get_nodes_per_face)(int* faceID, int* nNodes);

  /*!
   * \brief Get the physical position of solution points in given cell
   *
   * input: cellID, nNodes
   * output: xyz [size: nNodes x 3, row-major]
   */
  void (*get_receptor_nodes)(int* cellID, int* nNodes, double* xyz);

  /*!
   * \brief Get the physical position of flux points on given face
   *        For the artificial boundary method
   *
   * @param[in]  faceID  Face ID within current rank
   * @param[in]  nNodes  Number of nodes expected on face
   * @param[out] xyz     Coordinates of each point on face [nNodes x 3, row-major]
   */
  void (*get_face_nodes)(int* faceID, int* nNodes, double* xyz);

  double (*get_q_spt)(int cellID, int spt, int var);

  double (*get_grad_spt)(int cellID, int spt, int dim, int var);
  double& (*get_grad_fpt)(int faceID, int fpt, int dim, int var);

  double& (*get_q_fpt)(int ff, int spt, int var);

  double* (*get_q_spts)(int& ele_stride, int& spt_stride, int& var_stride, int etype);
  double* (*get_dq_spts)(int& ele_stride, int& spt_stride, int& var_stride, int& dim_stride, int etype);

  // GPU-related functions
  double* (*get_q_spts_d)(int etype);
  double* (*get_dq_spts_d)(int& ele_stride, int& spt_stride, int& var_stride, int& dim_stride, int etype);

  void (*get_face_nodes_gpu)(int* fringeIDs, int nFringe, int* nptPerFace, double* xyz);
  void (*get_cell_nodes_gpu)(int* cellIDs, int nCells, int* nptPerCell, double* xyz);

  int (*get_n_weights)(int cellID);
  void (*donor_frac_gpu)(int* donorIDs, int nFringe, double* rst, double* weights);

  /*! Copy updated solution/gradient for fringe faces from host to device */
  void (*face_data_to_device)(int* fringeIDs, int nFringe, int gradFlag, double *data);

  /*! Copy updated solution/gradient for fringe cells from host to device */
  void (*cell_data_to_device)(int* cellIDs, int nCells, int gradFlag, double *data);

  /*!
   * \brief Determine whether a point (x,y,z) lies within a cell
   *
   * Given a point's physical position, determine if it is contained in the
   * given cell; if so, return the reference coordinates of the point
   *
   * @param[in]  cellID    ID of cell within current mesh
   * @param[in]  xyz       Physical position of point to test
   * @param[out] passFlag  Is the point inside the cell? (no:0, yes:1)
   * @param[out] rst       Position of point within cell in reference coordinates
   */
  void (*donor_inclusion_test)(int* cellID, double* xyz, int* passFlag, double* rst);

  /*!
   * \brief Get interpolation weights for a point
   *
   * Get interpolation points & weights for current cell,
   * given a point in reference coordinates
   *
   * @param[in]  cellID    ID of cell within current mesh
   * @param[in]  xyz       Physical position of receptor point
   * @param[out] nweights  Number of interpolation points/weights to be used
   * @param[out] inode     Indices of donor points within global solution array
   * @param[out] weights   Interpolation weights for each donor point
   * @param[in]  rst       Reference coordinates of receptor point within cell
   * @param[in]  buffsize  Amount of memory allocated to 'weights' (# of doubles)
   */
  void (*donor_frac)(int* cellID, double* xyz, int* nweights, int* inode,
                     double* weights, double* rst, int* buffsize);

  void (*convert_to_modal)(int *,int *,double *,int *,int *,double *);

  int nreceptorCells;      /** number of receptor cells */
  int *ctag;               /** index of receptor cells */
  int *pointsPerCell;      /** number of receptor points per cell */
  int maxPointsPerCell;    /** max of pointsPerCell vector */

  /* ---- Artificial Boundary Variables ---- */
  int nreceptorFaces;      /** Number of artificial boundary faces */
  int *ftag;               /** Indices of artificial boundary faces */
  int *pointsPerFace;      /** number of receptor points per face */
  int maxPointsPerFace;    /** max of pointsPerFace vector */

  int nFacePoints;
  int nCellPoints;

  std::vector<double> rxyz;            /**  point coordinates */
  int ipoint; 
  int *picked;             /** < flag specifying if a node has been selected for high-order interpolation */

  int nreceptorCellsCart;
  int *ctag_cart;
  int *pickedCart;
 	
 public :
  int ntotalPointsCart;
  double *rxyzCart;
  int *donorIdCart;
  int donorListLength;

  int nfringe;
  int meshtag; /** < tag of the mesh that this block belongs to */
  double resolutionScale = 1.0;

  //! Oriented bounding box of this partition
  OBB *obb;

  //! Axis-aligned bounding box for this partition
  double aabb[6];

  int nsearch;        /** < number of query points to search in this block */
  std::vector<int> isearch;       /** < index of query points in the remote process */
  std::vector<double> xsearch;    /** < coordinates of the query points */
  std::vector<int> tagsearch;    /** < coordinates of the query points */
  std::vector<int> xtag;    /** < map to find duplicate points */
  std::vector<double> res_search;    /** < resolution of search points */
  std::vector<double> res_search0;    /** < resolution of search points */

#ifdef _GPU
  hvec<double> rst;
  hvec<double> rst_h; //! Specificallly for donor_frac_gpu
  hvec<int> donorId; /// TODO: allow hvec to be used for non-GPU cases (use malloc vs. cudaMalloc)
  hvec<int> donorId_h; //! Specificallly for donor_frac_gpu
#else
  std::vector<double> rst; /// TODO: allow hvec to be used for non-GPU cases (use malloc vs. cudaMalloc)
  std::vector<int> donorId;
#endif
  bool haveDonors = false;

  int donorCount;
  int myid,nproc;
  std::vector<double> cellRes;  /** < resolution for each cell */
  int ntotalPoints;        /**  total number of extra points to interpolate */
  int ihigh;               /** High-order flag for current rank */
  int iartbnd;             /** High-order artificial boundary flag for current rank */
  bool gpu = false;        /** Flag for GPUs being used on high-order solver */

  int ninterp2;            /** < number of interpolants for high-order points */
  int interp2ListSize;
  std::vector<INTERPLIST> interpList2; /** < list for high-interpolation points */
  int ninterpCart;
  int interpListCartSize;
  INTERPLIST *interpListCart; 

  // Direct-Cut Method Variables - TO BE CLEANED UP
  int nDims = 3;
  int nGroups;
  int nCutFringe, nCutHole;   //! # of fringe/hole-cutting faces on this rank
  int gridType;               //! Type of grid: background (0) or normal (1)
  std::vector<int> cutFacesW, cutFacesO; //! Wall and overset cut face lists
  std::vector<std::vector<int>> cutFaces;  //! List of faces on each cut group
  std::vector<int> groupIDs;
  std::set<int> myGroups;
  std::vector<int> cutType_glob;
  std::vector<int> nGf; //! Number of faces on each cutting group

  /* ---- GPU-Related Variables ---- */
  int nSpts;  // Number of spts per ele on this rank
#ifdef _GPU
  dvec<double> weights_d;
  hvec<double> weights_h;
  dvec<int> donors_d;
  dvec<int> donorsBT_d;
  dvec<char> etypes_d;
  dvec<int> nweights_d;
  dvec<int> winds_d;
  dvec<int> buf_inds_d;
  hvec<int> buf_inds;
  std::vector<int> buf_disp;

  dvec<double*> qptrs_d;
  dvec<int> qstrides_d;

  cudaStream_t stream_handle;
  cudaEvent_t event_handle;
#endif

  /* ---- Additional callbacks for p4est ---- */

  //! Call back functions to use p4est to search its own internal data
  void (*p4estsearchpt) (double *,int *,int *,int *);
  void (*check_intersect_p4est) (int *, int *);

  /** basic constructor */
  MeshBlock()
  {
    nv=NULL; nc=NULL; x=NULL;
    iblank=NULL; iblank_cell=NULL; iblank_face=NULL; vconn=NULL;
    wbcnode=NULL; obcnode=NULL;
    adt=NULL; obb=NULL;
    donorList=NULL; interp2donor=NULL;
    nsearch=0;
    cancelList=NULL;
    userSpecifiedNodeRes=NULL; userSpecifiedCellRes=NULL;
    nfringe=2;
    // new vars
    ninterp=ninterp2=interpListSize=interp2ListSize=0;
    ctag=NULL;
    ftag=NULL;
    pointsPerCell=NULL;
    pointsPerFace=NULL;
    maxPointsPerCell=0;
    ntotalPoints=0;
    nreceptorFaces=0;
    ihigh=0;
    ipoint=0;
    picked=NULL;
    ctag_cart=NULL;
    rxyzCart=NULL;
    donorIdCart=NULL;
    pickedCart=NULL;
    ntotalPointsCart=0;
    nreceptorCellsCart=0;
    ninterpCart=0;
    interpListCartSize=0;
    interpListCart=NULL;
  }

  /** basic destructor */
  ~MeshBlock();
      
  void preprocess(void);

  void updateOBB(void);

  void tagBoundary(void);
  
  void writeGridFile(int bid);

  void writeFlowFile(int bid,double *q,int nvar,int type);
  
  void setData(int btag, int nnodesi, double *xyzi, int *ibli, int nwbci, int nobci,
      int *wbcnodei, int *obcnodei, int ntypesi, int *nvi, int* ncfi, int *nci, int **vconni);

  void setFaceData(int _gtype, int _nftype, int* _nf, int* _nfv, int** _f2v,
      int *_f2c, int **_c2f, int* _ib_face, int nOver, int nWall, int nMpi, int* oFaces,
      int* wFaces, int* mFaces, int* procR, int* idR);

  void setResolutions(double *nres,double *cres);    

  void setCommMap(int ns, int nr, int *sm, int *rm)
  {
    nsend = ns;  nrecv = nr;
    sndMap.assign(sm, sm+nsend);
    rcvMap.assign(rm, rm+nrecv);
  }

  void setGridVelocity(double *grid_vel);

  void setTransform(double *mat, double* pvt, double* offset, int ndim);

  void calcNextGrid(double dt);
  void resetCurrentGrid(void);
  int getIterIblanks(void);
  void clearUnblanks(void);
  void swapPointers(void);

  void search();

  /*! Given a 3D position, find the cell it lies within (-1 if not found) */
  int findPointDonor(double *x_pt);

  /*! Given a bounding box, find all elements which overlap with it */
  std::unordered_set<int> findCellDonors(double *bbox);

  void writeOBB(int bid);

  void writeOBB2(OBB *obc,int bid);

  void updateSolnData(int inode,double *qvar,double *q,int nvar,int interptype);

  int getNinterp(void) {return ninterp;}

  void getInterpolatedSolution(int *nints,int *nreals,int **intData,double **realData,double *q,
			       int nvar, int interptype);

  void getInterpolatedSolution2(int &nints,int &nreals,int *&intData, double *&realData,
                                double *q,int nvar, int interptype);

  void getInterpolatedSolutionAMR(int *nints,int *nreals,int **intData,double **realData,double *q,
				  int nvar, int interptype);

  void getInterpolatedSolutionAtPointsAMR(int *nints,int *nreals,int **intData,double **realData,
					 double **q,
					 int nvar, int interptype);
  
  void checkContainment(int *cellIndex,int adtElement,double *xsearch,double *rst);

  void getWallBounds(int *mtag,int *existWall, double wbox[6]);

  void getOversetBounds(int *mtag,int *existOver, double obox[6]);
  
  void markWallBoundary(int *sam,int nx[3],double extents[6]);

  void markOversetBoundary(int *sam,int nx[3],double extents[6]);

  void getQueryPoints(OBB *obb,int *nints,int **intData,int *nreals,
		      double **realData);
  

  /** routines that do book keeping */

  void getDonorPacket(PACKET *sndPack, int nsend);

  void initializeDonorList();
  
  void insertAndSort(int pointid, int senderid, int meshtag, int remoteid, double donorRes);
  
  void processDonors(HOLEMAP *holemap, int nmesh,int **donorRecords,double **receptorResolution,
		     int *nrecords);

  void initializeInterpList(int ninterp_input);
  
  void findInterpData(int& recid, int irecord, double receptorRes);

  void findInterpListCart();

  void set_ninterp(int);

  void getCancellationData(int& nints, int*& intData);

  void cancelDonor(int irecord);

  void getInterpData(int& nrecords, int*& donorData);

  void clearIblanks(void);

  void getStats(int mstat[2]);

  void setIblanks(int inode);

  void getDonorCount(int *dcount,int *fcount);

  void getDonorInfo(int *receptors,int *indices, double *frac);

  void getReducedOBB(OBB *obc, double *points);

  void getReducedOBB_SuperMesh(OBB *obc, double *points);

  //
  // routines for high order connectivity and interpolation
  //
  void setAllCellsNormal(void);

  void getCellIblanks(const MPI_Comm meshComm);

  //! Find all artificial boundary faces using previously-set cell iblank values
  void calcFaceIblanks(const MPI_Comm &meshComm);

  void getCutGroupBoxes(std::vector<double> &cutBox, std::vector<std::vector<double>> &faceBox, int nGroups_glob);

  int getCuttingFaces(std::vector<double> &faceNodesW, std::vector<double> &faceNodesO, std::vector<double>& bboxW, std::vector<double>& bboxO);

  void getDirectCutCells(std::vector<std::unordered_set<int>> &cellList, std::vector<double> &cutBox_global, int nGroups_glob);

  //! Determine blanking status based upon given set of wall and overset faces
  void directCut(std::vector<double> &cutFaces, int nCut, int nvertf, std::vector<double> &cutBbox, CutMap& cutMap, int cutType = 1);

  //! Peform the Direct Cut alogorithm on the GPU
  void directCut_gpu(std::vector<double> &cutFaces, int nCut, int nvertf, std::vector<double>& cutBbox,
      HOLEMAP &holeMap, CutMap &cutMap, int cutType = 1);

  //! Take the union of all cut flags
  void unifyCutFlags(std::vector<CutMap> &cutMap);

  void set_cell_iblank(int *iblank_cell_input)
  {
    iblank_cell=iblank_cell_input;
  }
  void setcallback(void (*f1)(int*, int*),
		    void (*f2)(int *,int *,double *),
		    void (*f3)(int *,double *,int *,double *),
		    void (*f4)(int *,double *,int *,int *,double *,double *,int *),
		   void (*f5)(int *,int *,double *,int *,int*,double *))
  {
    get_nodes_per_cell=f1;
    get_receptor_nodes=f2;
    donor_inclusion_test=f3;
    donor_frac=f4;
    convert_to_modal=f5;

    ihigh = 1;
  }

  //! Set callback functions specific to Artificial Boundary method
  void setCallbackArtBnd(void (*gnf)(int* id, int* npf),
                         void (*gfn)(int* id, int* npf, double* xyz),
                         double (*gqs)(int ic, int spt, int var),
                         double& (*gqf)(int ff, int fpt, int var),
                         double (*ggs)(int ic, int spt, int dim, int var),
                         double& (*ggf)(int ff, int fpt, int dim, int var),
                         double* (*gqss)(int& es, int& ss, int& vs, int etype),
                         double* (*gdqs)(int& es, int& ss, int& vs, int& ds, int etype))
  {
    // See declaration of functions above for more details
    get_nodes_per_face = gnf;
    get_face_nodes = gfn;
    get_q_spt = gqs;
    get_q_fpt = gqf;
    get_grad_spt = ggs;
    get_grad_fpt = ggf;
    get_q_spts = gqss;
    get_dq_spts = gdqs;

    iartbnd = 1;
  }

  void setCallbackArtBndGpu(void (*h2df)(int* ids, int nf, int grad, double *data),
                            void (*h2dc)(int* ids, int nf, int grad, double *data),
                            double* (*gqd)(int etype),
                            double* (*gdqd)(int& es, int& ss, int& vs, int& ds, int etype),
                            void (*gfng)(int*, int, int*, double*),
                            void (*gcng)(int*, int, int*, double*),
                            int (*gnw)(int),
                            void (*dfg)(int*, int, double*, double*))
  {
    face_data_to_device = h2df; // this is the function to move face data to device
    cell_data_to_device = h2dc;
    get_q_spts_d = gqd;
    get_dq_spts_d = gdqd;
    get_face_nodes_gpu = gfng;
    get_cell_nodes_gpu = gcng;
    get_n_weights = gnw;
    donor_frac_gpu = dfg;
    gpu = true;
  }

  void setp4estcallback(void (*f1)(double *,int *,int *,int *),
      void (*f2)(int *,int *))
  {
    p4estsearchpt=f1;
    check_intersect_p4est=f2;
  }

  void writeCellFile(int, int* flag = NULL);

  /*! Gather a list of all receptor point locations (including for high-order) */
  void getInternalNodes(void);

  /*! Gather a list of all artificial boundary point locations (for high-order)
   * [Requires use of callback functions] */
  void getFringeNodes(bool unblanking = false);

  void getExtraQueryPoints(OBB *obb, int &nints, int*& intData, int &nreals,
                           double*& realData);
  void processPointDonors(void);

  void processPointDonorsGPU(void);

  void getInterpolatedSolutionAtPoints(int *nints, int *nreals, int **intData,
      double **realData, double *q, int nvar, int interpType);

  void getInterpolatedGradientAtPoints(int &nints, int &nreals, int *&intData,
      double *&realData, double *q, int nvar);

  /*! Update high-order element data at internal degrees of freedom */
  void updatePointData(double *q,double *qtmp,int nvar,int interptype);

  /*! Update high-order element data at artificial boundary flux points */
  void updateFringePointData(double *qtmp, int nvar);

  /*! Update solution gradient at artificial boundary flux points */
  void updateFringePointGradient(double *dqtmp, int nvar);

  /*! Copy fringe-face data back to device for computation in solver */
  void sendFringeDataGPU(int gradFlag);

  void outputOrphan(FILE *fp,int i) 
  {
    fprintf(fp,"%f %f %f\n",rxyz[3*i],rxyz[3*i+1],rxyz[3*i+2]);
  }

  void outputOrphan(std::ofstream &fp, int i)
  {
    fp << i << " " << rxyz[3*i] << " " << rxyz[3*i+1] << " " << rxyz[3*i+2] << std::endl;
  }

  void clearOrphans(HOLEMAP *holemap, int nmesh, int *itmp);
  void getUnresolvedMandatoryReceptors();
  void getCartReceptors(CartGrid *cg, parallelComm *pc, int itype=0);
  void setCartIblanks(void);

  void getIgbpData(double *& igbp_ptr);
  int getNIgbps(void) { return igbpdata.size()/4; }

  void setupADT(void);

  //! Rebuild the ADT using knowledge of current donor elements
  void rebuildADT(void);

  /*! Setup additional helpful connectivity structures */
  void extraConn(void);

  /* ---- GPU-Related Functions ---- */
#ifdef _GPU
  void setupBuffersGPU(int nsend, std::vector<int>& intData, std::vector<VPACKET>& sndPack);
  void interpSolution_gpu(double* q_out_d, int nvar);
  void interpGradient_gpu(double* dq_out_d, int nvar);

  void set_stream_handle(cudaStream_t handle, cudaEvent_t event);

  //! Set pointers to storage of geometry data on device
  void setDeviceData(double* xyz, double* coords, int* ibc, int* ibf);
#endif
  void set_soasz(unsigned int sz);
  unsigned int get_soasz() { return soasz;};

  void set_maxnface_maxnfpts(unsigned int, unsigned int);
  void set_face_fpts(int* ffpts, unsigned int ntface);   
  void set_fcelltypes(int* fctype, unsigned int ntface);
  void set_fposition(int* fpos, unsigned int ntface);
  void set_face_numbers(unsigned int nmpif, unsigned int nbcf);

  void set_data_reorder_map(int* srted, int* unsrted, int ncells);
  void set_interior_mapping(unsigned long long int basedata, 
                            unsigned long long int grad_basedata,
                            int* faceinfo, int* mapping, 
                            int* grad_mapping, int* grad_strides,
                            int nfpts);
  void set_interior_gradient_mapping();
  void set_mpi_rhs_mapping(unsigned long long int basedata, int* mapping, int* strides, int nfpts);
  void set_mpi_mapping(unsigned long long int basedata, int* faceinfo, int* mapping, int nfpts);
  void set_overset_rhs_basedata(unsigned long long int basedata);
  void set_overset_mapping(unsigned long long int basedata, int* faceinfo, int* mapping, int nfpts);
  void set_facecoords_mapping(unsigned long long int basedata, int* faceinfo, int* mapping, int nfpts);
  void figure_out_interior_artbnd_target();
  void figure_out_mpi_artbnd_target();
  void figure_out_overset_artbnd_target();
  void figure_out_facecoords_target();

  void prepare_interior_artbnd_target_data(double* data, int nvar);
  void prepare_interior_artbnd_target_data_gradient(double* data, int nvar, int dim);
  void prepare_mpi_artbnd_target_data(double* data, int nvar);
  void prepare_overset_artbnd_target_data(double* data, int nvar);

  void update_fringe_face_info(unsigned int flag);

  void reset_mpi_face_artbnd_status_pointwise(unsigned int nvar);
  void reset_entire_mpi_face_artbnd_status_pointwise(unsigned int nvar);

  void unpack_interior_artbnd_u_pointwise(unsigned int nvar);
  void unpack_interior_artbnd_du_pointwise(unsigned int nvar, unsigned int dim);
  
  void pack_fringe_facecoords_pointwise(double* rxyz);
  void set_cell_info_by_type(unsigned int nctypes, unsigned int nc,
                             int* ctypes, int* nupts_per_type,
                             int* ustrides, int* dustrides, unsigned long long* du_basedata,
                             int* cstrides, unsigned long long* c_basedata
                            );
  void pointwise_pack_cell_coords(int ntotal, double* rxyz);
  void pointwise_unpack_cell_soln(double* data, int nvar);
};

#endif
