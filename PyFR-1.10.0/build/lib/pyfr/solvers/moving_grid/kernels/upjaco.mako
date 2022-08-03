# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='upjaco' ndim='2'
              smats='inout fpdtype_t[${str(ndims)}][${str(ndims)}]'
              smatr='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='out fpdtype_t'
              rcpdjacr='in fpdtype_t'>

    // X_ξ  Y_ξ  Z_ξ
    // X_η  Y_η  Z_η
    // X_ζ  Y_ζ  Z_ζ

    fpdtype_t jaco[${ndims}][${ndims}];
% for j in range(ndims):
% for i in range(ndims):
    jaco[${i}][${j}] = smats[${i}][${j}];
% endfor
% endfor

% if ndims == 2:
    
    // col is the leading dimension
    smats[0][0] = jaco[1][1];  
    smats[0][1] =-jaco[1][0];

    smats[1][0] =-jaco[0][1];
    smats[1][1] = jaco[0][0];

    rcpdjac = 1.0/(jaco[0][0]*jaco[1][1]-jaco[0][1]*jaco[1][0]);
% elif ndims == 3:
    
    // col is the leading dimension
    smats[0][0] = jaco[1][1]*jaco[2][2]-jaco[1][2]*jaco[2][1];
    smats[0][1] = jaco[1][2]*jaco[2][0]-jaco[1][0]*jaco[2][2];
    smats[0][2] = jaco[1][0]*jaco[2][1]-jaco[1][1]*jaco[2][0];

    smats[1][0] = jaco[2][1]*jaco[0][2]-jaco[2][2]*jaco[0][1];
    smats[1][1] = jaco[2][2]*jaco[0][0]-jaco[2][0]*jaco[0][2];
    smats[1][2] = jaco[2][0]*jaco[0][1]-jaco[2][1]*jaco[0][0];

    smats[2][0] = jaco[0][1]*jaco[1][2]-jaco[0][2]*jaco[1][1];
    smats[2][1] = jaco[0][2]*jaco[1][0]-jaco[0][0]*jaco[1][2];
    smats[2][2] = jaco[0][0]*jaco[1][1]-jaco[0][1]*jaco[1][0];

    rcpdjac = jaco[0][0]*smats[0][0] + jaco[0][1]*smats[0][1] + jaco[0][2]*smats[0][2];
    rcpdjac = 1.0/rcpdjac;
% endif
</%pyfr:kernel>
