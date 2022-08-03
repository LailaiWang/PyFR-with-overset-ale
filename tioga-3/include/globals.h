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
 global definitions for
 the interface code
*/
tioga *tg;
/*
** pointer storage for connectivity arrays that
** comes from external solver
*/
int **vconn = NULL;  /// Cell-to-vertex connectivity
int *nc = NULL;      /// Number of cells (per cell type)
int *ncf = NULL;     /// Number of faces per cell (per cell type)
int *nv = NULL;      /// Number of vertices per cell (per cell type)

int **fconn = NULL;  /// face-to-vertex connectivity
int *nf = NULL;      /// Number of faces (per face type)
int *nfv = NULL;     /// Number of vertices per face (per face type)
int *a;
//int *celloffset=NULL;
