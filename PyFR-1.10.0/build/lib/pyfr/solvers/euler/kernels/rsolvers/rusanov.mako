# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, mvell, mvelr, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl', 'mvell')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr', 'mvelr')};
    
    fpdtype_t nv, a;

% if mvgrid is True:
    // Sum the left and right velocities and take the normal
    nv = ${pyfr.dot('n[{i}]','vl[{i}]-mvell[{i}][0] + vr[{i}]-mvelr[{i}][0]',i=ndims)};
% else:
    // Sum the left and right velocities and take the normal
    nv = ${pyfr.dot('n[{i}]', 'vl[{i}] + vr[{i}]', i=ndims)};
% endif
    // Estimate the maximum wave speed / 2
    a = sqrt(${0.25*c['gamma']}*(pl + pr)/(ul[0] + ur[0])) + 0.25*fabs(nv);

    // Output
% for i in range(nvars):
    nf[${i}] = 0.5*(${' + '.join('n[{j}]*(fl[{j}][{i}] + fr[{j}][{i}])'
                                 .format(i=i, j=j) for j in range(ndims))})
             + a*(ul[${i}] - ur[${i}]);
% endfor
</%pyfr:macro>
