# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% beta = c['ldg-beta'] %>

<%pyfr:kernel name='mpiconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in mpi fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              ovmarker='in view fpdtype_t'>
// an overset-mpi face has a 0.5 beta as pre-setup
// When a non-overset mpi face become artificial boundary, 
// we need to force ldg-beta to be 0.5, regardless what the possible value it is 
// such that it wouldn't matter for the other face

% if ovset is True:
//% if ovmarker > 0.0: 
//    beta = 0.5; // force ldg-beta to be 0.5
//% endif
% endif

% for i in range(nvars):
% if beta == -0.5:
    ulout[${i}] = ulin[${i}];
% elif beta == 0.5:
    ulout[${i}] = urin[${i}];
% else:
    ulout[${i}] = urin[${i}]*${0.5 + beta} + ulin[${i}]*${0.5 - beta};
% endif
% endfor

</%pyfr:kernel>

