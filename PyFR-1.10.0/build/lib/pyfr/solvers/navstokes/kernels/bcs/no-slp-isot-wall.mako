# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul,mvell,nl,ur,mvelr' externs='ploc, t'>
    ur[0] = ul[0];
% if mvgrid is True:
    
    // copy
% for i in range(ndims):
    mvelr[${i}][0] = mvell[${i}][0];
% endfor
    
    // add grid velocity
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = -ul[${i + 1}] + 2*${c[v]}*ul[0]+ul[0]*mvell[${i}][0];
% endfor

% else:

% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = -ul[${i + 1}] + 2*${c[v]}*ul[0];
% endfor

% endif
    ur[${nvars - 1}] = ${c['cpTw']/c['gamma']}*ur[0]
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_state' params='ul,mvell,nl,ur,mvelr' externs='ploc, t'>
    ur[0] = ul[0];

% if mvgrid is True:

% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = ${c[v]}*ul[0]+ul[0]*mvell[${i}][0];
% endfor

% else:

% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = ${c[v]}*ul[0];
% endfor

% endif

    ur[${nvars - 1}] = ${c['cpTw']/c['gamma']}*ur[0]
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>
