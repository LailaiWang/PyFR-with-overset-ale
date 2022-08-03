# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% one = 1 %>

<%pyfr:kernel name='upmvelface' ndim='2'
              t='scalar fpdtype_t',
              omegax = 'scalar fpdtype_t'
              omegay = 'scalar fpdtype_t'
              omegaz = 'scalar fpdtype_t'
              pivotx = 'scalar fpdtype_t'
              pivoty = 'scalar fpdtype_t'
              pivotz = 'scalar fpdtype_t'
              tvelx  = 'scalar fpdtype_t'
              tvely  = 'scalar fpdtype_t'
              tvelz  = 'scalar fpdtype_t'
              ploc='in fpdtype_t[${str(ndims)}]'
              pref='in fpdtype_t[${str(ndims)}]'
              mvel='inout fpdtype_t[${str(ndims)}][${str(one)}]'>
    
% for i in range(ndims):
    mvel[${i}][0] = 0.0;
% endfor

% if ndims == 3:
    fpdtype_t x = ploc[0], y = ploc[1], z = ploc[2];
    fpdtype_t rx = x-pivotx, ry = y-pivoty, rz = z-pivotz;

    fpdtype_t lrx = omegay*rz-omegaz*ry;
    fpdtype_t lry = omegaz*rx-omegax*rz;
    fpdtype_t lrz = omegax*ry-omegay*rx;

    mvel[0][0] = tvelx + lrx;
    mvel[1][0] = tvely + lry;
    mvel[2][0] = tvelz + lrz;
% else:
    fpdtype_t x = ploc[0], y = ploc[1], z = 0.0;
    fpdtype_t rx = x-pivotx, ry = y-pivoty, rz = z-pivotz;

    fpdtype_t lrx = omegay*rz-omegaz*ry;
    fpdtype_t lry = omegaz*rx-omegax*rz;

    mvel[0][0] = tvelx + lrx;
    mvel[1][0] = tvely + lry;
% endif

</%pyfr:kernel>
