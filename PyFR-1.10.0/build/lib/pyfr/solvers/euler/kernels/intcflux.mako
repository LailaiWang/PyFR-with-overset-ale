# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<% one = 1%>

<%pyfr:kernel name='intcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              mvelnl='out view fpdtype_t'
              mvelnr='out view fpdtype_t'             
              mvell='in view fpdtype_t[${str(ndims)}][${str(one)}]'
              mvelr='in view fpdtype_t[${str(ndims)}][${str(one)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'>
    // Perform the Riemann solve

    fpdtype_t fn[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'mvell', 'mvelr', 'nl', 'fn')};

    // Scale and write out the common normal fluxes
% for i in range(nvars):
    ul[${i}] =  magnl*fn[${i}];
    ur[${i}] = -magnl*fn[${i}];
% endfor

% if mvgrid is True:
    mvelnl =  ${pyfr.dot('mvell[{i}][0]+mvelr[{i}][0]','nl[{i}]',i=ndims)};
    mvelnl = 0.5*magnl*mvelnl;
    mvelnr = -mvelnl; 
%endif
</%pyfr:kernel>
