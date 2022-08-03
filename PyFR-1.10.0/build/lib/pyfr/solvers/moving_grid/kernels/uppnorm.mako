# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name  ='uppnorm' ndim='1'
              snorm ='in fpdtype_t[${str(ndims)}]'
              smat  ='in view fpdtype_t[${str(ndims)}][${str(ndims)}]'
              magn  ='out fpdtype_t'
              norm  ='out fpdtype_t[${str(ndims)}]' >
    
    fpdtype_t rcpmag;
% for i in range(ndims):
    norm[${i}] =${' + '.join('smat[{j}][{i}]*snorm[{j}]'
                             .format(i=i,j=j) for j in range(ndims))};
% endfor
    
    magn = ${' + '.join('norm[{j}]*norm[{j}]'.format(j=j) for j in range(ndims))};
    magn = sqrt(magn);
    rcpmag = 1.0/magn;

//normalize norm
% for i in range(ndims):
    norm[${i}] = norm[${i}]*rcpmag;
% endfor

</%pyfr:kernel>
