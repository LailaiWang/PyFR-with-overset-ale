# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

% if bccfluxstate:
<%include file='pyfr.solvers.navstokes.kernels.bcs.${bccfluxstate}'/>
% endif

<% one = 1 %>

<%pyfr:kernel name='bccflux' ndim='1'
              ul='inout view fpdtype_t[${str(nvars)}]'
              gradul='in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
              artviscl='in view fpdtype_t'
              mvelnl = 'out view fpdtype_t'
              mvell='in view fpdtype_t[${str(ndims)}][${str(one)}]'
              nl='in fpdtype_t[${str(ndims)}]'
              magnl='in fpdtype_t'>
    ${pyfr.expand('bc_common_flux_state','ul','gradul','artviscl','mvell','nl','magnl')};

    //printf(" %15.7e %15.7e %15.7e %15.7e %15.7e\n", ul[0], ul[1], ul[2], ul[3], ul[4]);
</%pyfr:kernel>
