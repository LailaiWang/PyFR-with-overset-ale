# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% lbeta, tau = c['ldg-beta'], c['ldg-tau'] %>

<%one = 1 %>

<%pyfr:kernel name='mpicflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              ur='inout mpi fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              gradur='in mpi fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artviscl='in view fpdtype_t'
              artviscr='in mpi fpdtype_t'
              mvelnl = 'out view fpdtype_t'
              mvelnr = 'out mpi fpdtype_t'
              mvell = 'in view fpdtype_t[${str(ndims)}][${str(one)}]'
              mvelr = 'inout mpi  fpdtype_t[${str(ndims)}][${str(one)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'
              ovmarker='in view fpdtype_t'>

    fpdtype_t beta;
    beta = ${lbeta};
// copy mvell into mvelr for mpi-overset interfaces
% if mvgrid is True:
% for i in range(ndims):
    mvelr[${i}][0] = mvell[${i}][0];
% endfor
% endif

% if ovmpi is True:
    %if lbeta !=0.5:
        printf("some thing is wrong, mpi-overset face beta != 0.5 %lf\n", beta);
    %endif
% else:
    %if overset is True:
        beta = ovmarker > 0? 0.5:beta;
    %endif
% endif

    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve','ul','ur','mvell','mvelr','nl','ficomm')};

    // move these out
    fpdtype_t fvl[${ndims}][${nvars}] = {{0}};
% if lbeta != -0.5:
    ${pyfr.expand('viscous_flux_add', 'ul', 'gradul', 'fvl')};
    ${pyfr.expand('artificial_viscosity_add', 'gradul', 'fvl', 'artviscl')};
% endif

    // move these out
    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
% if lbeta != 0.5:
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', 'gradur', 'fvr', 'artviscr')};
% endif

% for i in range(nvars):
    // here use beta to check, downgrade to c language
    if (beta == -0.5) {
        printf("detecting negative beta\n");
        fvcomm = ${' + '.join('nl[{j}]*fvr[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
    } else if (beta == 0.5) {
        printf("detecting positive beta\n");
        fvcomm = ${' + '.join('nl[{j}]*fvl[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
    } else {
        fvcomm = (0.5 + beta)*(${' + '.join('nl[{j}]*fvl[{j}][{i}]'
                                         .format(i=i, j=j)
                                         for j in range(ndims))})
               + (0.5 - beta)*(${' + '.join('nl[{j}]*fvr[{j}][{i}]'
                                         .format(i=i, j=j)
                                         for j in range(ndims))});
    }

% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif

    ul[${i}] = magnl*(ficomm[${i}] + fvcomm);
% endfor
% if mvgrid is True:
    mvelnl = ${pyfr.dot('mvell[{i}][0]+mvelr[{i}][0]','nl[{i}]',i=ndims)};
    mvelnl = 0.5*magnl*mvelnl;
% endif
</%pyfr:kernel>
