# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% lbeta = c['ldg-beta'] %>
<% one = 1 %>

<%pyfr:kernel name='mpiconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in mpi fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              ovmarker='in view fpdtype_t'>

// an overset-mpi face has a 0.5 beta as pre-setup
// When a non-overset mpi face become artificial boundary, 
// we need to force ldg-beta to be 0.5, regardless what the possible value it is 
// such that it wouldn't matter for the other face

    fpdtype_t beta;
    beta = ${lbeta};
% if ovmpi is True:
    % if lbeta != 0.5:
        printf("some thing is wrong, mpi-overset face beta != 0.5 %lf\n", beta);
    % endif
% else:
    %if overset is True:
        beta = ovmarker > 0? 0.5:beta;
    %endif
% endif

% for i in range(nvars):
    if (beta == - 0.5) {
        ulout[${i}] = ulin[${i}];
    } else if ( beta == 0.5) {
        ulout[${i}] = urin[${i}];
    } else {
        ulout[${i}] = urin[${i}]*(0.5 + beta) + ulin[${i}]*(0.5 - beta);
    }
% endfor

</%pyfr:kernel>

