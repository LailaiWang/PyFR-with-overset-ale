# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% beta, tau = c['ldg-beta'], c['ldg-tau'] %>

<% one = 1%>

<%pyfr:kernel name='intcflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradur='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artviscl='in view fpdtype_t'
              artviscr='in view fpdtype_t'
              mvelnl='out view fpdtype_t'
              mvelnr='out view fpdtype_t'
              mvell='in view fpdtype_t[${str(ndims)}][${str(one)}]'
              mvelr='in view fpdtype_t[${str(ndims)}][${str(one)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'>

    // Perform the Riemann solve
    // integrating grid velocity
    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve', 'ul', 'ur','mvell','mvelr', 'nl', 'ficomm')};

% if beta != -0.5:
    fpdtype_t fvl[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ul', 'gradul', 'fvl')};
    ${pyfr.expand('artificial_viscosity_add', 'gradul', 'fvl', 'artviscl')};
% endif

% if beta != 0.5:
    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', 'gradur', 'fvr', 'artviscr')};
% endif

% for i in range(nvars):
% if beta == -0.5:
    fvcomm = ${' + '.join('nl[{j}]*fvr[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
% elif beta == 0.5:
    fvcomm = ${' + '.join('nl[{j}]*fvl[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
% else:
    fvcomm = ${0.5 + beta}*(${' + '.join('nl[{j}]*fvl[{j}][{i}]'
                                         .format(i=i, j=j)
                                         for j in range(ndims))})
           + ${0.5 - beta}*(${' + '.join('nl[{j}]*fvr[{j}][{i}]'
                                         .format(i=i, j=j)
                                         for j in range(ndims))});
% endif
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif

    ul[${i}] =  magnl*(ficomm[${i}] + fvcomm);
    ur[${i}] = -magnl*(ficomm[${i}] + fvcomm);
% endfor
    // simply taking average here for the common flux of GCL
% if mvgrid is True:
    mvelnl =  ${pyfr.dot('mvell[{i}][0]+mvelr[{i}][0]','nl[{i}]',i=ndims)};
    mvelnl = 0.5*magnl*mvelnl;
    mvelnr = -mvelnl; 
% endif
</%pyfr:kernel>
