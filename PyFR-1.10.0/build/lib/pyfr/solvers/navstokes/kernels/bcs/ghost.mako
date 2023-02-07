<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

<% tau = c['ldg-tau'] %>
<% one =1 %>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, mvell, nl, magnl'>
    // Viscous states // Allocate mvelr no matter if mvell exists or not
    fpdtype_t ur[${nvars}], gradur[${ndims}][${nvars}], mvelr[${ndims}][${one}];
    ${pyfr.expand('bc_ldg_state','ul','mvell','nl','ur','mvelr')};
    ${pyfr.expand('bc_ldg_grad_state', 'ul', 'nl', 'gradul', 'gradur')};

    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', 'gradur', 'fvr', 'artviscl')};

    // Inviscid (Riemann solve) state
    ${pyfr.expand('bc_rsolve_state','ul','mvell','nl','ur','mvelr')};

    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve', 'ul', 'ur','mvell','mvelr','nl', 'ficomm')};

% for i in range(nvars):
    fvcomm = ${' + '.join('nl[{j}]*fvr[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif

    ul[${i}] = magnl*(ficomm[${i}] + fvcomm);
% endfor
% if mvgrid is True:
    mvelnl = ${pyfr.dot('mvell[{i}][0]','nl[{i}]',i=ndims)};
    mvelnl = mvelnl*magnl;
% endif
</%pyfr:macro>
