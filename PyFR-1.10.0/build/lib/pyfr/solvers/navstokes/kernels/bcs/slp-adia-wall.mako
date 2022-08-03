# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<%pyfr:macro name='bc_ldg_state' params='ul,mvell, nl, ur, mvelr'>
    fpdtype_t nor = ${' + '.join('ul[{1}]*nl[{0}]'.format(i, i + 1)
                                 for i in range(ndims))};
    ur[0] = ul[0];
% if mvgrid is True:
    
    // copy
% for i in range(ndims):
    mvelr[${i}][0] = mvell[${i}][0];
% endfor
    //
% for i in range(ndims):
    ur[${i + 1}] = ul[${i + 1}] - 2*nor*nl[${i}] + ul[0]*mvel[${i}][0];
% endfor

% else:

% for i in range(ndims):
    ur[${i + 1}] = ul[${i + 1}] - 2*nor*nl[${i}];
% endfor

% endif
    ur[${nvars - 1}] = ul[${nvars - 1}];

</%pyfr:macro>

<%one = 1%>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl,mvell, nl, magnl'>
    // Ghost state r
    fpdtype_t ur[${nvars}], mvelr[${ndims}][${one}];
    ${pyfr.expand('bc_ldg_state','ul','mvell','nl','ur','mvelr')};
    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}];
    ${pyfr.expand('rsolve','ul','ur','mvell','mvelr','nl','ficomm')};

% for i in range(nvars):
    ul[${i}] = magnl*(ficomm[${i}]);
% endfor
</%pyfr:macro>
