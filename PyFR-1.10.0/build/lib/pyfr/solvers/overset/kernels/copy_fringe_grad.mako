# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr: kernel name = 'copy_fringe_grad' ndim ='1'
               gradfpts = 'in view fpdtype_t[${str(ndims)}][${str(nvars)}]'
               fringe = 'out fpdtype_t [${str(ndims)}][${str(nvars)}]'>

% for i,j in pyfr.ndrange(ndims, nvars):
    fringe[${i}][${j}] = gradfpts[${i}][${j}];  
% endfor

</%pyfr:kernel>
