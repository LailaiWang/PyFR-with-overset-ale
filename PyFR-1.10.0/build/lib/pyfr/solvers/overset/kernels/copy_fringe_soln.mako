# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr: kernel name = 'copy_fringe_soln' ndim ='1'
               solnfpts = 'in view fpdtype_t[${str(nvars)}]'
               fringe = 'out fpdtype_t [${str(nvars)}]'>

% for i in range(nvars):
    fringe[${i}] = solnfpts[${i}];  
% endfor

</%pyfr:kernel>
