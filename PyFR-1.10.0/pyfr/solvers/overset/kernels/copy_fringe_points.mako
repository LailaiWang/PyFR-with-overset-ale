# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr: kernel name = 'copy_fringe_points' ndim ='1'
               coordfpts = 'in view fpdtype_t[${str(ndims)}]'
               fringe = 'out fpdtype_t[${str(ndims)}]'>

% for i in range(ndims):
    fringe[${i}] = coordfpts[${i}];  
% endfor

</%pyfr:kernel>
