# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='rotjaco' ndim='2'
              r00 = 'scalar fpdtype_t'
              r01 = 'scalar fpdtype_t'
              r02 = 'scalar fpdtype_t'
              r10 = 'scalar fpdtype_t'
              r11 = 'scalar fpdtype_t'
              r12 = 'scalar fpdtype_t'
              r20 = 'scalar fpdtype_t'
              r21 = 'scalar fpdtype_t'
              r22 = 'scalar fpdtype_t'
              smats='inout fpdtype_t[${str(ndims)}][${str(ndims)}]'
              smatr='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='out fpdtype_t'
              rcpdjacr='in fpdtype_t'>

    // X_ξ  Y_ξ  Z_ξ
    // X_η  Y_η  Z_η
    // X_ζ  Y_ζ  Z_ζ

    fpdtype_t jaco[${ndims}][${ndims}], jacorot[${ndims}][${ndims}];
% for j in range(ndims):
% for i in range(ndims):
    jaco[${i}][${j}] = smatr[${i}][${j}];
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
    
    // inverse of smatr here

    jaco[0][0] = smatr[1][1]*smatr[2][2]-smatr[1][2]*smatr[2][1];
    jaco[0][1] = smatr[1][2]*smatr[2][0]-smatr[1][0]*smatr[2][2];
    jaco[0][2] = smatr[1][0]*smatr[2][1]-smatr[1][1]*smatr[2][0];

    jaco[1][0] = smatr[2][1]*smatr[0][2]-smatr[2][2]*smatr[0][1];
    jaco[1][1] = smatr[2][2]*smatr[0][0]-smatr[2][0]*smatr[0][2];
    jaco[1][2] = smatr[2][0]*smatr[0][1]-smatr[2][1]*smatr[0][0];

    jaco[2][0] = smatr[0][1]*smatr[1][2]-smatr[0][2]*smatr[1][1];
    jaco[2][1] = smatr[0][2]*smatr[1][0]-smatr[0][0]*smatr[1][2];
    jaco[2][2] = smatr[0][0]*smatr[1][1]-smatr[0][1]*smatr[1][0];
    
    // apply rotation matrix

    // smatr*R R here the rotation matrix's inverse

    jacorot[0][0] = jaco[0][0]*r00 + jaco[0][1]*r01 + jaco[0][2]*r02;
    jacorot[1][0] = jaco[0][0]*r10 + jaco[0][1]*r11 + jaco[0][2]*r12;
    jacorot[2][0] = jaco[0][0]*r20 + jaco[0][1]*r21 + jaco[0][2]*r22;

    jacorot[0][1] = jaco[1][0]*r00 + jaco[1][1]*r01 + jaco[1][2]*r02;
    jacorot[1][1] = jaco[1][0]*r10 + jaco[1][1]*r11 + jaco[1][2]*r12;
    jacorot[2][1] = jaco[1][0]*r20 + jaco[1][1]*r21 + jaco[1][2]*r22;

    jacorot[0][2] = jaco[2][0]*r00 + jaco[2][1]*r01 + jaco[2][2]*r02;
    jacorot[1][2] = jaco[2][0]*r10 + jaco[2][1]*r11 + jaco[2][2]*r12;
    jacorot[2][2] = jaco[2][0]*r20 + jaco[2][1]*r21 + jaco[2][2]*r22;

    smats[0][0] = jacorot[1][1]*jacorot[2][2]-jacorot[1][2]*jacorot[2][1];
    smats[0][1] = jacorot[1][2]*jacorot[2][0]-jacorot[1][0]*jacorot[2][2];
    smats[0][2] = jacorot[1][0]*jacorot[2][1]-jacorot[1][1]*jacorot[2][0];

    smats[1][0] = jacorot[2][1]*jacorot[0][2]-jacorot[2][2]*jacorot[0][1];
    smats[1][1] = jacorot[2][2]*jacorot[0][0]-jacorot[2][0]*jacorot[0][2];
    smats[1][2] = jacorot[2][0]*jacorot[0][1]-jacorot[2][1]*jacorot[0][0];

    smats[2][0] = jacorot[0][1]*jacorot[1][2]-jacorot[0][2]*jacorot[1][1];
    smats[2][1] = jacorot[0][2]*jacorot[1][0]-jacorot[0][0]*jacorot[1][2];
    smats[2][2] = jacorot[0][0]*jacorot[1][1]-jacorot[0][1]*jacorot[1][0];

    rcpdjac = rcpdjacr;
% endif
</%pyfr:kernel>
