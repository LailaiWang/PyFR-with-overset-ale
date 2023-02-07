# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>

<%one = 1 %>

<%pyfr:kernel name='bcconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              mvell='in view fpdtype_t[${str(ndims)}][${str(one)}]'
              nlin='in fpdtype_t[${str(ndims)}]'>

    fpdtype_t mvelr[${ndims}][${one}];
    ${pyfr.expand('bc_ldg_state','ulin','mvell','nlin','ulout','mvelr')};
</%pyfr:kernel>
