# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='upplocface' ndim='2'
              t='scalar fpdtype_t'
              r00 = 'scalar fpdtype_t'
              r01 = 'scalar fpdtype_t'
              r02 = 'scalar fpdtype_t'
              r10 = 'scalar fpdtype_t'
              r11 = 'scalar fpdtype_t'
              r12 = 'scalar fpdtype_t'
              r20 = 'scalar fpdtype_t'
              r21 = 'scalar fpdtype_t'
              r22 = 'scalar fpdtype_t'
              ofx = 'scalar fpdtype_t'
              ofy = 'scalar fpdtype_t'
              ofz = 'scalar fpdtype_t'
              pvx = 'scalar fpdtype_t'
              pvy = 'scalar fpdtype_t'
              pvz = 'scalar fpdtype_t'
              ploc='inout fpdtype_t[${str(mvars)}]'
              pref='inout fpdtype_t[${str(mvars)}]'
              >

% if ndims == 2:
    ploc[0] = pref[0]+ofx;
    ploc[1] = pref[1]+ofy;
% else: 
    // apply offset to the reference coordinates
    ploc[0] = pref[0]+ofx;
    ploc[1] = pref[1]+ofy;
    ploc[2] = pref[2]+ofz;

    pvx = pvx + ofx;
    pvy = pvy + ofy;
    pvz = pvz + ofz;

    //printf("pvx %lf %lf %lf\n", pvx, pvy, pvz);
    
    fpdtype_t x0 = 0.0, x1 = 0.0, x2 = 0.0;
    
    // apply  rotation
    x0 = r00*(ploc[0]-pvx)+r01*(ploc[1]-pvy)+r02*(ploc[2]-pvz);
    x1 = r10*(ploc[0]-pvx)+r11*(ploc[1]-pvy)+r12*(ploc[2]-pvz);
    x2 = r20*(ploc[0]-pvx)+r21*(ploc[1]-pvy)+r22*(ploc[2]-pvz);

    ploc[0] = x0+pvx;
    ploc[1] = x1+pvy;
    ploc[2] = x2+pvz;

% endif

</%pyfr:kernel>
