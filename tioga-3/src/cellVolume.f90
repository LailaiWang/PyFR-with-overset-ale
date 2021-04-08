! This file is part of the Tioga software library

! Tioga  is a tool for overset grid assembly on parallel distributed systems
! Copyright (C) 2015 Jay Sitaraman

! This library is free software; you can redistribute it and/or
! modify it under the terms of the GNU Lesser General Public
! License as published by the Free Software Foundation; either
! version 2.1 of the License, or (at your option) any later version.

! This library is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
! Lesser General Public License for more details.

! You should have received a copy of the GNU Lesser General Public
! License along with this library; if not, write to the Free Software
! Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
!=============================================================!
subroutine cellVolume(vol,xc,numverts,fconn,nfaces,nvert)
!
!    
!     Subroutine to find volumes of arbitrary polyhedra
!     with triangles and quads as faces
!     uses vertex coordinates and provided face connectivity
!
!     Uses Gausss divergence theorem with quad faces treated
!     as bilinear
!     reference Appendix B of
!     https://drum.umd.edu/dspace/handle/1903/310
!     Sitaraman, UMD, PhD thesis 2003
!
!     Arguments:
!
!     xc - coordinates of vertices of the polyhedra
!     fconn - face connectivity
!     numverts-number of vertices in each face
!     nfaces - number of faces
!     nvert - number of vertices of the polyhedra     
!
!     Last updated by j.sitaraman 12/14/2007
!
!=============================================================!
!
! subroutine arguments
!
implicit none
real*8, intent(inout) :: vol   !< cell volume
integer, intent(in) :: nvert   !< number of vertices
integer, intent(in) :: nfaces  !< number of faces
real*8, intent(in)  :: xc(3,nvert) !< coordinates of each vertex
integer, intent(in) :: numverts(nfaces) !< number of vertices for each face
integer, intent(in) :: fconn(4,nfaces)  !< connectivity for each face (triangles have fconn(4,i)=0)
real*8  :: scalarProduct
!
! local variables
!
integer :: iface
!
! begin
!
vol=0.
!
! -ve sign below because the vertex ordering of faces are such
! that the normals are inward facing
!
do iface=1,nfaces
 if (numverts(iface)==3) then
    vol=vol-0.5* scalarProduct(xc(:,fconn(1,iface)), &
                                             xc(:,fconn(2,iface)), &
                                             xc(:,fconn(3,iface)))
 else
    vol=vol-0.25*scalarProduct(xc(:,fconn(1,iface)), &
                                             xc(:,fconn(2,iface)), &
                                             xc(:,fconn(3,iface)))

    vol=vol-0.25*scalarProduct(xc(:,fconn(1,iface)), &
                                             xc(:,fconn(3,iface)), &
                                             xc(:,fconn(4,iface)))

    vol=vol-0.25*scalarProduct(xc(:,fconn(1,iface)), &
                                             xc(:,fconn(2,iface)), &
                                             xc(:,fconn(4,iface)))

    vol=vol-0.25*scalarProduct(xc(:,fconn(2,iface)), &
                                             xc(:,fconn(3,iface)), &
                                             xc(:,fconn(4,iface)))
 endif
enddo
!
vol=vol/3.0
!
return
end subroutine cellVolume

!==========================================================================
! box product of vectors, a,b,c
!==========================================================================
function scalarProduct(a,b,c)
implicit none
real*8 scalarProduct
real*8 a(3),b(3),c(3)

scalarProduct = a(1)*b(2)*c(3) - a(1)*b(3)*c(2) &
               +a(2)*b(3)*c(1) - a(2)*b(1)*c(3) & 
               +a(3)*b(1)*c(2) - a(3)*b(2)*c(1)

return
end function scalarProduct
