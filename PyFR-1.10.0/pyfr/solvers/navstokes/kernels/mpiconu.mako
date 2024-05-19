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
% if ovset is True:

% if ovmaker is not UNDEFINED:
    printf("ovmarker is %lf\n", ovmarker);

% if ovmarker > 0.0: 
    beta = 0.5; // force ldg-beta to be 0.5
% endif

% endif

% endif

% for i in range(nvars):
% if beta == -0.5:
    ulout[${i}] = ulin[${i}];
% elif beta == 0.5:
    ulout[${i}] = urin[${i}];
% else:
    ulout[${i}] = urin[${i}]*(0.5 + beta) + ulin[${i}]*(0.5 - beta);
% endif
% endfor

</%pyfr:kernel>

